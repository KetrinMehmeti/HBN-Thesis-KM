#!/usr/bin/env python3
"""Runtime entrypoint applying the generic botocore not-found handler."""

import sys

from botocore.exceptions import ClientError

import consolidate_embeddings as pipeline


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False
        raise


pipeline.object_exists = object_exists


if __name__ == "__main__":
    sys.exit(pipeline.main())
