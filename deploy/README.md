# Deploy templates

`ecs-task-definition.example.json` is a **redacted** Fargate task definition.

- Replace `ACCOUNT_ID`, `REGION`, `TAG` and role names yourself.
- Put every secret in AWS Secrets Manager (or SSM) and reference it from the `secrets` array.
- Never commit a real `task-definition.json` or `service.json`. Both names are gitignored.
- Set `API_KEY`. Leave ERP posting flags false until you intend to write to an ERP.
- `CORS_ORIGINS` should be an explicit allow-list, not `*`, in any shared environment.
