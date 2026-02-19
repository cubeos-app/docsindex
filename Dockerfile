# Build stage
FROM golang:1.21-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app
COPY go.mod ./
COPY *.go ./

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o cubeos-docsindex .

# Runtime stage
FROM alpine:3.19

RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /app
COPY --from=builder /app/cubeos-docsindex .

RUN mkdir -p /cubeos/docs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -q --spider http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/cubeos-docsindex"]
