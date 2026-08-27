FROM golang:1.25-alpine AS builder

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o gateway .

FROM alpine:3.24

# Needed for the HTTPS JWKS fetch; the binary is static otherwise.
RUN apk --no-cache add ca-certificates
WORKDIR /app

COPY --from=builder /build/gateway .

# nobody: the process listens on 8080 and writes nothing, so it never needs root.
USER 65534:65534
EXPOSE 8080
CMD ["./gateway"]
