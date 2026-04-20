"""
AWS CDK Stack for Odoo RAG automated data pipeline.

Creates:
1. S3 bucket for data storage
2. Lambda function for pipeline execution
3. S3 event trigger on sources.json upload
4. IAM roles and permissions
"""

from aws_cdk import (
    Stack,
    Duration,
    CfnOutput,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_s3_notifications as s3n,
    aws_iam as iam,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_ecs_patterns as ecs_patterns,
    RemovalPolicy,
)
from constructs import Construct


class OdooRagPipelineStack(Stack):
    """CDK Stack for automated RAG data pipeline."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 bucket for data storage
        data_bucket = s3.Bucket(
            self,
            "OdooRagDataBucket",
            bucket_name="odoo-rag-data-2026-cohort-v2",
            versioned=True,  # Keep version history
            removal_policy=RemovalPolicy.RETAIN,  # Don't delete on stack destroy
            auto_delete_objects=False,
        )

        # Lambda function for pipeline
        # Note: CDK automatically manages ECR repository (cdk-hnb659fds-container-assets-...)
        pipeline_function = lambda_.DockerImageFunction(
            self,
            "OdooRagPipelineFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                directory="../",  # Root of project
                file="infrastructure/Dockerfile.pipeline",
                platform=ecr_assets.Platform.LINUX_AMD64,  # Lambda x86_64
            ),
            memory_size=2048,  # 2GB for processing
            timeout=Duration.minutes(15),  # Max Lambda timeout
            environment={
                "S3_BUCKET": data_bucket.bucket_name,
                "S3_INPUT_PREFIX": "input/",
                "S3_OUTPUT_PREFIX": "output/",
                "HF_HOME": "/tmp/huggingface",  # HuggingFace cache in writable /tmp
            },
        )

        # Grant Lambda permissions to read/write S3
        data_bucket.grant_read_write(pipeline_function)

        # Grant Lambda permissions to call Bedrock (for QA generation)
        pipeline_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["*"],  # Bedrock models don't have specific ARNs
            )
        )

        # S3 event notification: trigger Lambda on sources.json upload
        data_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(pipeline_function),
            s3.NotificationKeyFilter(prefix="input/", suffix="sources.json"),
        )

        # Outputs
        self.data_bucket = data_bucket  # Export bucket object for app stack
        self.data_bucket_name = data_bucket.bucket_name
        self.pipeline_function_name = pipeline_function.function_name


class OdooRagAppStack(Stack):
    """CDK Stack for Gradio app deployment (Lambda or ECS)."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        data_bucket_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Reference existing data bucket
        data_bucket = s3.Bucket.from_bucket_name(self, "DataBucket", data_bucket_name)

        # Use default VPC
        vpc = ec2.Vpc.from_lookup(self, "VPC", is_default=True)

        # Create ECS cluster
        cluster = ecs.Cluster(self, "OdooRagCluster", vpc=vpc)

        # Create Fargate service with ALB
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "OdooRagService",
            cluster=cluster,
            cpu=2048,  # 2 vCPU
            memory_limit_mib=4096,  # 4 GB
            desired_count=1,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset(
                    "../",
                    file="Dockerfile",
                    platform=ecr_assets.Platform.LINUX_AMD64,  # Force x86_64 for ECS
                ),
                container_port=7860,
                environment={
                    "S3_BUCKET": data_bucket_name,
                    "S3_PREFIX": "output/",
                },
            ),
            public_load_balancer=True,
            # Force tasks to run in public subnets with public IP for ECR access
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            assign_public_ip=True,
        )

        # Grant S3 read access to task role
        data_bucket.grant_read(fargate_service.task_definition.task_role)

        # Grant Bedrock permissions to task role
        fargate_service.task_definition.task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["*"],
            )
        )

        # Configure health check
        fargate_service.target_group.configure_health_check(
            path="/",
            interval=Duration.seconds(60),
            timeout=Duration.seconds(30),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        # Output the ALB URL
        self.app_url = fargate_service.load_balancer.load_balancer_dns_name
        CfnOutput(
            self,
            "AppUrl",
            value=f"http://{fargate_service.load_balancer.load_balancer_dns_name}",
            description="Public URL for the Gradio app (ALB)",
        )
