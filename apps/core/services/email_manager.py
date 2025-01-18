import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from pytest_is_running import is_running

from apps.accounts.services.token import TokenService
from config.settings import EmailServiceConfig, AppConfig


class EmailService:
    config = EmailServiceConfig.get_config()
    app = AppConfig.get_config()

    @classmethod
    def __send_email(cls, subject: str, body: str, to_address: str):
        try:
            message = MIMEMultipart()
            message['From'] = cls.config.smtp_username
            message['To'] = to_address
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))

            # Connect to the SMTP server
            with smtplib.SMTP(cls.config.smtp_server, cls.config.smtp_port) as server:
                server.set_debuglevel(1)  # Optional: useful for debugging
                server.starttls()  # Upgrade connection to secure
                server.login(cls.config.smtp_username, cls.config.smtp_password)
                server.sendmail(cls.config.smtp_username, to_address, message.as_string())
                # server.quit() is not needed when using 'with' block
        except Exception as e:
            print(f"An error occurred while sending email: {e}")

    @classmethod
    def __print_test_otp(cls, otp: str):
        dev_show = f"--- Testing OTP: {otp} ---"
        print(dev_show)

    @classmethod
    def __send_verification_email(cls, subject, body, to_address):
        """
        Sends a verification email or prints OTP in testing mode.
        """

        if is_running() or cls.config.use_local_fallback:
            cls.__print_test_otp(TokenService.create_otp_token())
        else:
            cls.__send_email(subject, body, to_address)

    @classmethod
    def register_send_verification_email(cls, to_address):
        """
        Sends a verification email for the registration process.
        """

        otp = TokenService.create_otp_token()
        print('otp ==> ',otp)
        subject = 'Email Verification'
        body = f"Thank you for registering with {cls.app.app_name}!\n\n" \
               f"To complete your registration, please enter the following code: {otp}\n\n" \
               f"If you didn't register, please ignore this email."
        cls.__send_verification_email(subject, body, to_address)

    @classmethod
    def reset_password_send_verification_email(cls, to_address):
        """
        Sends a verification email for the password reset process.
        """

        otp = TokenService.create_otp_token()
        subject = 'Password Reset Verification'
        body = f"We received a request to reset your {cls.app.app_name} password.\n\n" \
               f"Please enter the following code to reset your password: {otp}\n\n" \
               f"If you didn't request this, you can ignore this email."
        cls.__send_verification_email(subject, body, to_address)

    @classmethod
    def change_email_send_verification_email(cls, new_email: str):
        """
        Sends a verification email for the email change process.
        """

        otp = TokenService.create_otp_token()
        subject = 'Email Change Verification'
        body = f"We received a request to change the email associated with your {cls.app.app_name} account.\n\n" \
               f"To confirm this change, please enter the following code: {otp}\n\n" \
               f"If you didn't request this, please contact our support team."
        cls.__send_verification_email(subject, body, new_email)
