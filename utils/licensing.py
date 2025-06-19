"""LEXA Licensing System

This module handles license validation, usage tracking, and subscription management.
"""

import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LicenseInfo:
    """License information container"""
    tier: str
    valid_until: datetime
    char_limit: int
    features: list
    organization: str
    is_valid: bool
    error: Optional[str] = None

class LicenseManager:
    """Manages LEXA licensing and subscription features"""
    
    TIERS = {
        'basic': {
            'char_limit': 10_000,
            'features': ['basic_analysis', 'export_basic'],
            'price': 0
        },
        'professional': {
            'char_limit': 100_000,
            'features': [
                'basic_analysis',
                'advanced_analysis',
                'export_all',
                'priority_support',
                'custom_metrics'
            ],
            'price': 99
        },
        'enterprise': {
            'char_limit': -1,  # Unlimited
            'features': [
                'basic_analysis',
                'advanced_analysis',
                'export_all',
                'priority_support',
                'custom_metrics',
                'api_access',
                'white_label',
                'custom_integration'
            ],
            'price': 'custom'
        }
    }

    def __init__(self, secret_key: str):
        """Initialize the license manager.
        
        Args:
            secret_key: JWT secret key for license validation
        """
        self.secret_key = secret_key

    def validate_license(self, license_key: str) -> LicenseInfo:
        """Validate a license key and return license information.
        
        Args:
            license_key: The license key to validate
            
        Returns:
            LicenseInfo object containing license details
        """
        try:
            # Decode and verify the license key
            payload = jwt.decode(
                license_key,
                self.secret_key,
                algorithms=['HS256']
            )
            
            # Check if license has expired
            expiry = datetime.fromtimestamp(payload['exp'])
            if expiry < datetime.now():
                return LicenseInfo(
                    tier='invalid',
                    valid_until=expiry,
                    char_limit=0,
                    features=[],
                    organization=payload.get('org', ''),
                    is_valid=False,
                    error='License has expired'
                )
            
            # Get tier information
            tier = payload['tier']
            if tier not in self.TIERS:
                return LicenseInfo(
                    tier='invalid',
                    valid_until=expiry,
                    char_limit=0,
                    features=[],
                    organization=payload.get('org', ''),
                    is_valid=False,
                    error='Invalid license tier'
                )
            
            # Return valid license info
            return LicenseInfo(
                tier=tier,
                valid_until=expiry,
                char_limit=self.TIERS[tier]['char_limit'],
                features=self.TIERS[tier]['features'],
                organization=payload.get('org', ''),
                is_valid=True
            )
            
        except jwt.ExpiredSignatureError:
            return LicenseInfo(
                tier='invalid',
                valid_until=datetime.now(),
                char_limit=0,
                features=[],
                organization='',
                is_valid=False,
                error='License has expired'
            )
        except jwt.InvalidTokenError:
            return LicenseInfo(
                tier='invalid',
                valid_until=datetime.now(),
                char_limit=0,
                features=[],
                organization='',
                is_valid=False,
                error='Invalid license key'
            )

    def generate_license(
        self,
        tier: str,
        organization: str,
        duration_days: int = 365
    ) -> str:
        """Generate a new license key.
        
        Args:
            tier: License tier ('basic', 'professional', 'enterprise')
            organization: Organization name
            duration_days: License validity in days
            
        Returns:
            JWT license key
        """
        if tier not in self.TIERS:
            raise ValueError(f"Invalid tier: {tier}")
        
        payload = {
            'tier': tier,
            'org': organization,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(days=duration_days)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def check_feature_access(self, license_info: LicenseInfo, feature: str) -> bool:
        """Check if the license has access to a specific feature.
        
        Args:
            license_info: LicenseInfo object
            feature: Feature to check
            
        Returns:
            bool indicating if feature is available
        """
        return license_info.is_valid and feature in license_info.features

    def track_usage(self, user_id: str, char_count: int) -> Dict:
        """Track character usage for a user.
        
        Args:
            user_id: User identifier
            char_count: Number of characters to track
            
        Returns:
            Dict with usage information
        """
        # In a production environment, this would update a database
        # For now, we'll just return a mock response
        return {
            'user_id': user_id,
            'timestamp': time.time(),
            'char_count': char_count,
            'status': 'tracked'
        }

    def get_usage_stats(self, user_id: str) -> Dict:
        """Get usage statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with usage statistics
        """
        # In a production environment, this would query a database
        # For now, we'll return mock data
        return {
            'user_id': user_id,
            'current_period': {
                'start': datetime.now() - timedelta(days=30),
                'end': datetime.now(),
                'chars_used': 5000,
                'chars_remaining': 5000
            },
            'history': [
                {
                    'date': datetime.now() - timedelta(days=i),
                    'chars_used': 100 * i
                } for i in range(30)
            ]
        }

    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool indicating if request should be allowed
        """
        # In a production environment, this would check a rate limiting system
        # For now, we'll always return True
        return True
