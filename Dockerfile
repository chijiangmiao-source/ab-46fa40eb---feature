FROM python:3.11-slim

# No third-party dependencies are required: the service uses the Python
# standard library only.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8080

WORKDIR /srv

COPY app/ ./app/

RUN adduser --system --no-create-home --uid 10001 appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=5 \
  CMD python -c "import os,sys,urllib.request as u; port=os.environ.get('API_PORT','8080'); r=u.urlopen('http://127.0.0.1:'+port+'/healthz', timeout=2); sys.exit(0 if r.status==200 else 1)"

CMD ["python", "-m", "app.server"]
