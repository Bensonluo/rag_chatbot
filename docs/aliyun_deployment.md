# Aliyun Deployment Guide

This guide explains how to deploy the RAG chatbot to Aliyun (Alibaba Cloud) ECS.

## Prerequisites

- Aliyun ECS instance (2 vCPU, 4GB RAM minimum)
- Domain name (optional, for HTTPS)
- Basic knowledge of Linux commands

## Step 1: Prepare ECS Instance

### 1.1 Create ECS Instance

```bash
# Recommended configuration:
# - Instance type: ecs.t6-c1m2.large (2 vCPU, 4GB RAM)
# - OS: Ubuntu 22.04 or CentOS 8
# - Storage: 40GB SSD
# - Network: VPC with public IP
```

### 1.2 Connect to ECS

```bash
ssh root@your-ecs-ip
# Or with key:
ssh -i your-key.pem root@your-ecs-ip
```

### 1.2 Install Docker

**Ubuntu/Debian:**
```bash
# Update packages
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (optional)
sudo usermod -aG docker $USER
```

**CentOS/RHEL:**
```bash
# Install Docker
sudo yum install -y docker

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### 1.3 Install Docker Compose

```bash
# Download Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

## Step 2: Deploy Application

### 2.1 Clone Repository

```bash
# Install Git
sudo apt-get install -y git  # Ubuntu/Debian
sudo yum install -y git      # CentOS/RHEL

# Clone repository
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2.2 Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Production .env configuration:**
```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql+asyncpg://postgres:your-password@postgres:5432/ragchatbot

# Redis
REDIS_URL=redis://redis:6379/0

# Vector DB
VECTOR_DB_URL=http://qdrant:6333
VECTOR_COLLECTION_NAME=documents

# LLM (choose one)
GLM_API_KEY=your-glm-api-key
GLM_MODEL=glm-4.5-air

# Embeddings
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-m3-v2-zh
EMBEDDING_DEVICE=cpu

# Security (IMPORTANT: Change this!)
SECRET_KEY=$(openssl rand -hex 32)

# CORS (update with your domain)
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
```

### 2.3 Deploy with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 2.4 Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Expected output:
# {
#   "status": "healthy",
#   "environment": "production"
# }
```

## Step 3: Configure Domain and SSL (Optional)

### 3.1 Set Up Domain

**Option A: Use Aliyun DNS**
1. Log in to Aliyun Console
2. Go to "Domain" → "DNS Management"
3. Add A record pointing to your ECS IP

**Option B: Use Cloudflare**
1. Sign up at Cloudflare
2. Add your domain
3. Change nameservers to Cloudflare
4. Add A record pointing to your ECS IP

### 3.2 Configure Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt-get install -y nginx

# Create config
sudo nano /etc/nginx/sites-available/rag-chatbot
```

**Nginx configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/rag-chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 3.3 Enable SSL with Certbot

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Certbot will automatically configure Nginx with HTTPS
```

## Step 4: Production Configuration

### 4.1 Set Up Firewall

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow SSH
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

### 4.2 Configure Automatic Backups

```bash
# Create backup script
cat > /root/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup"

# Backup volumes
docker run --rm \
  --volumes-from rag-chatbot-db \
  -v $BACKUP_DIR:/backup \
  ubuntu tar czf /backup/postgres_$DATE.tar.gz /var/lib/postgresql/data

docker run --rm \
  --volumes-from rag-chatbot-qdrant \
  -v $BACKUP_DIR:/backup \
  ubuntu tar czf /backup/qdrant_$DATE.tar.gz /qdrant/storage

# Keep last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x /root/backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add: 0 2 * * * /root/backup.sh
```

### 4.3 Monitor Resources

```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check Docker stats
docker stats

# View logs
docker-compose logs -f --tail=100 api
```

## Step 5: Scaling and Optimization

### 5.1 Scale API Service

```bash
# Run multiple API instances
docker-compose up -d --scale api=3

# Configure load balancer (Nginx example):
upstream api_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://api_backend;
    }
}
```

### 5.2 Optimize Database

```bash
# Connect to PostgreSQL
docker exec -it rag-chatbot-db psql -U postgres -d rag_chatbot

# Create indexes
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);

# Vacuum and analyze
VACUUM ANALYZE;
```

### 5.3 Monitor Performance

```bash
# Enable monitoring profile
docker-compose --profile monitoring up -d

# Access Grafana at http://your-ecs-ip:3001
# Default credentials: admin/admin
```

## Troubleshooting

### Service won't start

```bash
# Check logs
docker-compose logs api

# Check resource usage
docker stats

# Restart services
docker-compose restart api
```

### Out of memory

```bash
# Add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Database connection issues

```bash
# Check PostgreSQL status
docker exec rag-chatbot-db pg_isready -U postgres

# View PostgreSQL logs
docker logs rag-chatbot-db
```

## Cost Estimation

**Monthly costs (approximate):**
- ECS (2 vCPU, 4GB): ~¥150-200/month
- Storage (40GB): ~¥10/month
- Bandwidth (first 5GB free): varies
- Domain (optional): ~¥50/year

**Total: ~¥200-250/month ($30-40 USD)**

## Security Checklist

- [ ] Change default SECRET_KEY
- [ ] Use strong database password
- [ ] Enable firewall (UFW)
- [ ] Set up SSL/HTTPS
- [ ] Regular security updates
- [ ] Configure automatic backups
- [ ] Monitor logs regularly
- [ ] Use environment variables for secrets
- [ ] Restrict SSH access (key-based only)
- [ ] Keep Docker images updated

## Maintenance

### Daily
- Check service health
- Monitor error logs

### Weekly
- Review resource usage
- Check disk space
- Review backup logs

### Monthly
- Update Docker images
- Security updates
- Review and rotate secrets

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Review this guide
3. Check GitHub issues
4. Contact support

## Next Steps

- Set up CI/CD pipeline
- Configure monitoring alerts
- Set up log aggregation
- Implement disaster recovery
- Performance tuning

---

**Congratulations!** Your RAG chatbot is now running on Aliyun! 🎉
