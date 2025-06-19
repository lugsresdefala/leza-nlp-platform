# LEXA - Deployment Guide

## Overview

LEXA (Linguistic Exploration and Advanced Analysis) is a professional text analysis platform that offers sophisticated linguistic analysis through computational metrics. This guide covers deployment and setup for enterprise customers.

## System Requirements

- Python 3.10 or higher
- PostgreSQL 13+ for production database
- 4GB RAM minimum (8GB recommended)
- 2 CPU cores minimum (4 cores recommended)

## Installation

1. Install the package using pip:
```bash
pip install lexa-nlp
```

2. Set up environment variables:
```bash
# Database Configuration
LEXA_DB_HOST=your_db_host
LEXA_DB_PORT=5432
LEXA_DB_NAME=lexa_prod
LEXA_DB_USER=your_db_user
LEXA_DB_PASSWORD=your_db_password

# Authentication
LEXA_JWT_SECRET=your_jwt_secret
LEXA_LICENSE_KEY=your_license_key

# Application Settings
LEXA_ENV=production
LEXA_LOG_LEVEL=INFO
```

3. Initialize the database:
```bash
python -m lexa.utils.init_db
```

4. Download required NLP models:
```bash
python -m spacy download pt_core_news_lg
python -m spacy download en_core_web_lg
```

## Deployment Options

### 1. Docker Deployment (Recommended)

```bash
# Pull the official image
docker pull lexanlp/lexa:latest

# Run with environment variables
docker run -d \
  --name lexa \
  -p 8000:8000 \
  -e LEXA_DB_HOST=your_db_host \
  -e LEXA_DB_PORT=5432 \
  -e LEXA_DB_NAME=lexa_prod \
  -e LEXA_DB_USER=your_db_user \
  -e LEXA_DB_PASSWORD=your_db_password \
  -e LEXA_JWT_SECRET=your_jwt_secret \
  -e LEXA_LICENSE_KEY=your_license_key \
  lexanlp/lexa:latest
```

### 2. Manual Deployment

```bash
# Clone the repository
git clone https://github.com/your-org/lexa.git
cd lexa

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py --server.port 8000
```

## Subscription Plans

LEXA offers three subscription tiers:

1. **Basic** (Free Trial)
   - 10,000 characters/month
   - Basic text analysis
   - Standard support

2. **Professional** ($99/month)
   - 100,000 characters/month
   - Advanced analysis features
   - Priority support
   - Custom metrics dashboard

3. **Enterprise** (Custom pricing)
   - Unlimited characters
   - Full feature access
   - Dedicated support
   - Custom integration
   - On-premise deployment option

## Security Considerations

1. **Database Security**
   - Use strong passwords
   - Enable SSL for database connections
   - Implement proper backup strategies
   - Regular security updates

2. **Application Security**
   - Keep dependencies updated
   - Enable HTTPS
   - Implement rate limiting
   - Regular security audits

3. **Authentication**
   - Use secure JWT tokens
   - Implement proper session management
   - Enable 2FA for admin accounts

## Monitoring and Maintenance

1. **Performance Monitoring**
   - Monitor system resources
   - Track API response times
   - Monitor database performance
   - Set up alerts for critical issues

2. **Regular Maintenance**
   - Database optimization
   - Log rotation
   - Backup verification
   - Security patches

## Support and Documentation

- Technical documentation: docs.lexanlp.com
- Support email: support@lexanlp.com
- Emergency contact: emergency@lexanlp.com

## License Management

The license key provided with your subscription validates:
- Subscription tier
- Usage limits
- Feature access
- Deployment restrictions

Contact sales@lexanlp.com for license upgrades or custom enterprise solutions.

## Troubleshooting

Common issues and solutions:

1. **Database Connection Issues**
   - Verify database credentials
   - Check network connectivity
   - Ensure proper SSL configuration

2. **Performance Issues**
   - Check system resources
   - Monitor database queries
   - Review application logs

3. **Authentication Issues**
   - Verify JWT configuration
   - Check license key validity
   - Review user permissions

## Updates and Upgrades

1. **Automatic Updates**
```bash
pip install --upgrade lexa-nlp
```

2. **Manual Updates**
```bash
git pull origin main
pip install -r requirements.txt
python -m lexa.utils.migrate_db
```

## Backup and Recovery

1. **Database Backup**
```bash
pg_dump -U your_db_user lexa_prod > backup.sql
```

2. **Database Restore**
```bash
psql -U your_db_user lexa_prod < backup.sql
```

## Additional Resources

- API Documentation
- Integration Guides
- Best Practices
- Performance Tuning
- Security Guidelines
