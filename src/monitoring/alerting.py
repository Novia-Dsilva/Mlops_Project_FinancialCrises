"""
Alerting System - Slack and Email Notifications

Sends alerts when pipeline issues are detected.
"""

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
from pathlib import Path


def load_config():
    """Load configuration from params.yaml."""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def send_slack_alert(title: str, message: str):
    """
    Send Slack notification.
    
    Args:
        title: Alert title
        message: Alert message
    """
    config = load_config()
    webhook_url = config.get('alerts', {}).get('slack_webhook')
    
    if not webhook_url or 'YOUR_WEBHOOK' in webhook_url:
        print(f"⚠️  Slack webhook not configured - skipping Slack alert")
        print(f"   {title}: {message}")
        return
    
    payload = {
        "text": f"*{title}*",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            }
        ]
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            print(f"✓ Slack alert sent: {title}")
        else:
            print(f"✗ Slack alert failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Slack alert error: {e}")


def send_email_alert(subject: str, body: str):
    """
    Send email notification.
    
    Args:
        subject: Email subject
        body: Email body
    """
    config = load_config()
    recipients = config.get('alerts', {}).get('email_recipients', [])
    
    if not recipients or 'example.com' in recipients[0]:
        print(f"⚠️  Email not configured - skipping email alert")
        print(f"   Subject: {subject}")
        return
    
    # TODO: Configure SMTP settings in params.yaml
    print(f"📧 Email alert: {subject}")
    # Implement email sending here