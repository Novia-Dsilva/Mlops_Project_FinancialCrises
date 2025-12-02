# """
# Alerting System - Slack and Email Notifications

# Sends alerts when pipeline issues are detected.
# """

# import requests
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import yaml
# from pathlib import Path


# def load_config():
#     """Load configuration from params.yaml."""
#     with open('params.yaml', 'r') as f:
#         return yaml.safe_load(f)


# # def send_slack_alert(title: str, message: str):
# #     """
# #     Send Slack notification.
    
# #     Args:
# #         title: Alert title
# #         message: Alert message
# #     """
# #     config = load_config()
# #     webhook_url = config.get('alerts', {}).get('slack_webhook')
    
# #     if not webhook_url or 'YOUR_WEBHOOK' in webhook_url:
# #         print(f"⚠️  Slack webhook not configured - skipping Slack alert")
# #         print(f"   {title}: {message}")
# #         return
    
# #     payload = {
# #         "text": f"*{title}*",
# #         "blocks": [
# #             {
# #                 "type": "header",
# #                 "text": {
# #                     "type": "plain_text",
# #                     "text": title
# #                 }
# #             },
# #             {
# #                 "type": "section",
# #                 "text": {
# #                     "type": "mrkdwn",
# #                     "text": message
# #                 }
# #             }
# #         ]
# #     }
    
# #     try:
# #         response = requests.post(webhook_url, json=payload)
# #         if response.status_code == 200:
# #             print(f"✓ Slack alert sent: {title}")
# #         else:
# #             print(f"✗ Slack alert failed: {response.status_code}")
# #     except Exception as e:
# #         print(f"✗ Slack alert error: {e}")


# def send_email_alert(subject: str, body: str):
#     """
#     Send email notification.
    
#     Args:
#         subject: Email subject
#         body: Email body
#     """
#     config = load_config()
#     recipients = config.get('alerts', {}).get('email_recipients', [])
    
#     if not recipients or 'example.com' in recipients[0]:
#         print(f"⚠️  Email not configured - skipping email alert")
#         print(f"   Subject: {subject}")
#         return
    
#     # TODO: Configure SMTP settings in params.yaml
#     print(f"📧 Email alert: {subject}")
#     # Implement email sending here



"""
Alerting System - Email Notifications

Sends email alerts when pipeline issues are detected.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from params.yaml."""
    config_path = Path('params.yaml')
    if not config_path.exists():
        logger.warning("params.yaml not found, using default config")
        return {
            'alerts': {
                'email_recipients': ['finance.stress.analyser@gmail.com'],
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': 'finance.stress.analyser@gmail.com'
            }
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class AlertManager:
    """
    Centralized alert management system.
    Handles email notifications for pipeline events.
    """
    
    def __init__(self):
        self.config = load_config()
        self.alerts_config = self.config.get('alerts', {})
    
    def send_alert(self, message: str, severity: str = 'INFO', 
                   component: str = 'pipeline', alert_type: str = 'GENERAL'):
        """
        Send alert via email.
        
        Args:
            message: Alert message
            severity: Alert severity (INFO, WARNING, ERROR, CRITICAL)
            component: Component that triggered alert
            alert_type: Type of alert (VALIDATION_FAILURE, PIPELINE_FAILURE, etc.)
        """
        subject = f"[{severity}] {alert_type}: {component}"
        self.send_email_alert(subject, message)
    
    def send_email_alert(self, subject: str, body: str):
        """
        Send email notification using Gmail SMTP.
        
        Args:
            subject: Email subject
            body: Email body
        """
        recipients = self.alerts_config.get('email_recipients', ['finance.stress.analyser@gmail.com'])
        sender_email = self.alerts_config.get('sender_email', 'finance.stress.analyser@gmail.com')
        smtp_server = self.alerts_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = self.alerts_config.get('smtp_port', 587)
        
        # Get app password from environment or config
        app_password = os.environ.get('GMAIL_APP_PASSWORD') or \
                      self.alerts_config.get('gmail_app_password', '')
        
        if not app_password:
            logger.warning("⚠️ Gmail app password not configured - skipping email alert")
            logger.info(f"   Subject: {subject}")
            logger.info(f"   Body preview: {body[:200]}...")
            return
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            
            # Create HTML version
            html_body = f"""
            <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                        .header {{ background-color: #f0f0f0; padding: 20px; border-left: 5px solid #007bff; }}
                        .content {{ padding: 20px; }}
                        .footer {{ padding: 20px; font-size: 12px; color: #666; }}
                        .critical {{ border-left-color: #dc3545; }}
                        .warning {{ border-left-color: #ffc107; }}
                        .info {{ border-left-color: #17a2b8; }}
                        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    <div class="header {'critical' if 'CRITICAL' in subject else 'warning' if 'WARNING' in subject else 'info'}">
                        <h2>{subject}</h2>
                    </div>
                    <div class="content">
                        <pre>{body}</pre>
                    </div>
                    <div class="footer">
                        <p>Financial Crisis Detection Pipeline Alert System</p>
                        <p>Timestamp: {self._get_timestamp()}</p>
                    </div>
                </body>
            </html>
            """
            
            # Attach both plain text and HTML versions
            part1 = MIMEText(body, 'plain')
            part2 = MIMEText(html_body, 'html')
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, app_password)
                server.send_message(msg)
            
            logger.info(f"✅ Email alert sent to {', '.join(recipients)}")
            logger.info(f"   Subject: {subject}")
            
        except smtplib.SMTPAuthenticationError:
            logger.error("❌ Gmail authentication failed - check app password")
            logger.error("   Generate app password at: https://myaccount.google.com/apppasswords")
        except Exception as e:
            logger.error(f"❌ Email alert failed: {e}")
    
    def _get_timestamp(self):
        """Get current timestamp for alerts."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


# Backward compatibility function
def send_email_alert(subject: str, body: str):
    """Legacy function - use AlertManager instead."""
    manager = AlertManager()
    manager.send_email_alert(subject, body)


# Test function
def test_alerts():
    """Test the alert system."""
    logger.info("\n" + "="*80)
    logger.info("TESTING ALERT SYSTEM")
    logger.info("="*80)
    
    manager = AlertManager()
    
    # Test email
    test_message = """
    This is a test alert from the Financial Crisis Detection Pipeline.
    
    System Status: ✅ Operational
    Components: All systems functioning normally
    
    This is an automated test message.
    """
    
    manager.send_alert(
        message=test_message,
        severity='INFO',
        component='AlertSystem',
        alert_type='TEST'
    )
    
    logger.info("\n✅ Alert test complete!")
    logger.info("Check your email at: finance.stress.analyser@gmail.com")


if __name__ == "__main__":
    test_alerts()