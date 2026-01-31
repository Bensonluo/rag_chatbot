# API Quick Start Guide

## Authentication Endpoints

### 1. Register a New User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123",
    "full_name": "John Doe"
  }'
```

**Response** (200 OK):
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_admin": false,
  "created_at": "2024-01-29T12:00:00Z"
}
```

### 2. Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123"
  }'
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### 3. Get Current User

```bash
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response** (200 OK):
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_admin": false,
  "created_at": "2024-01-29T12:00:00Z"
}
```

### 4. Refresh Access Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```

**Response** (200 OK):
```json
{
  "access_token": "new_access_token_here...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

## Session Endpoints

### 1. Create a New Session

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Chat Session",
    "memory_type": "sliding_window",
    "context_window": 15
  }'
```

**Response** (200 OK):
```json
{
  "id": 1,
  "user_id": 1,
  "title": "My Chat Session",
  "memory_type": "sliding_window",
  "context_window": 15,
  "created_at": "2024-01-29T12:00:00Z",
  "updated_at": "2024-01-29T12:00:00Z",
  "message_count": 0
}
```

**Memory Types**:
- `sliding_window` - Keep last N messages
- `summarization` - Summarize old messages
- `hybrid` - Combine both strategies

### 2. List User Sessions

```bash
curl -X GET "http://localhost:8000/api/v1/sessions?page=1&page_size=20" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response** (200 OK):
```json
{
  "items": [
    {
      "id": 1,
      "user_id": 1,
      "title": "My Chat Session",
      "memory_type": "sliding_window",
      "context_window": 15,
      "created_at": "2024-01-29T12:00:00Z",
      "updated_at": "2024-01-29T12:00:00Z",
      "message_count": 0
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

### 3. Get a Specific Session

```bash
curl -X GET http://localhost:8000/api/v1/sessions/1 \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response** (200 OK):
```json
{
  "id": 1,
  "user_id": 1,
  "title": "My Chat Session",
  "memory_type": "sliding_window",
  "context_window": 15,
  "created_at": "2024-01-29T12:00:00Z",
  "updated_at": "2024-01-29T12:00:00Z",
  "message_count": 0
}
```

### 4. Update a Session

```bash
curl -X PUT http://localhost:8000/api/v1/sessions/1 \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Updated Title",
    "context_window": 20
  }'
```

**Response** (200 OK):
```json
{
  "id": 1,
  "user_id": 1,
  "title": "Updated Title",
  "memory_type": "sliding_window",
  "context_window": 20,
  "created_at": "2024-01-29T12:00:00Z",
  "updated_at": "2024-01-29T12:01:00Z",
  "message_count": 0
}
```

### 5. Delete a Session

```bash
curl -X DELETE http://localhost:8000/api/v1/sessions/1 \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response** (204 No Content)

## Error Responses

### 400 Bad Request - Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "password"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 401 Unauthorized - Invalid Token

```json
{
  "detail": "Invalid authentication credentials"
}
```

### 401 Unauthorized - No Token

```json
{
  "detail": "Not authenticated"
}
```

### 403 Forbidden - Inactive User

```json
{
  "detail": "User account is inactive"
}
```

### 404 Not Found

```json
{
  "detail": "Session not found: 999"
}
```

### 409 Conflict - Duplicate Email

```json
{
  "detail": "User with email 'user@example.com' already exists"
}
```

### 422 Unprocessable Entity - Weak Password

```json
{
  "detail": "Password validation failed: Password must be at least 8 characters long, Password must contain at least one uppercase letter"
}
```

## Using the Interactive API Docs

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide:
- Interactive API exploration
- Try-out functionality
- Request/response schemas
- Authentication support (click "Authorize" button)

## Authentication Flow

1. **Register** a new user account
2. **Login** to get access and refresh tokens
3. **Use access token** in Authorization header for authenticated requests
4. **Refresh token** when access token expires (30 minutes default)
5. **Login again** if refresh token expires (7 days default)

## Password Requirements

- Minimum 8 characters
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one digit (0-9)

## Session Configuration

- **title**: Session name (optional, max 255 chars)
- **memory_type**: Memory strategy
  - `sliding_window` (default)
  - `summarization`
  - `hybrid`
- **context_window**: Messages to keep in context (1-100, default 10)
