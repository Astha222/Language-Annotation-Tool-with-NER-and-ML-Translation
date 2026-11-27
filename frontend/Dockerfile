# ==========================================
# Stage 1: Build the static Expo Web bundle
# ==========================================
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY package*.json ./

# Install frontend dependencies
RUN npm install

# Copy application source code
COPY . .

# Build argument to configure backend API base URL
ARG EXPO_PUBLIC_API_URL=http://localhost:8000
ENV EXPO_PUBLIC_API_URL=$EXPO_PUBLIC_API_URL

# Export static web bundle to /app/dist
RUN npx expo export -p web

# ==========================================
# Stage 2: Serve static bundle via Nginx
# ==========================================
FROM nginx:alpine

# Remove default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy exported web distribution from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose web application port
EXPOSE 8081

CMD ["nginx", "-g", "daemon off;"]
