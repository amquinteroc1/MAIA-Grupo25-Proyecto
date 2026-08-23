# DVC/S3 setup pending

The project currently uses a local DVC remote while the AWS Academy bucket and
temporary credentials are pending. Credentials must never be committed to Git,
`.dvc/config`, or any report artifact.

## Current local remote

```bash
source ~/env-dvc/bin/activate
cd ~/dvc-proj
dvc remote list
```

Expected project remote during this phase:

```text
local-remote   /tmp/dvcstore   (default)
```

## Later S3 activation

After the team supplies a real bucket name and confirms the AWS Academy
session, execute interactively on the VM:

```bash
source ~/env-dvc/bin/activate
aws --version
dvc remote add -d s3-remote s3://REAL-BUCKET-NAME
dvc push
```

Do not run `aws configure` through an automated command and do not place
`aws_access_key_id`, `aws_secret_access_key`, or `aws_session_token` in the
repository. Validate the bucket and the DVC objects from the AWS console or
with `aws s3 ls` after the upload.
