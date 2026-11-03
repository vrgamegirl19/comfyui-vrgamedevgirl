# LLM API certificate errors

`CERTIFICATE_VERIFY_FAILED: certificate has expired` means HTTPS certificate validation failed. The API has not reached model generation or API-key validation at that point. The certificate can belong to the API service or an HTTPS-inspection proxy, and an incorrect computer clock can also make a certificate appear expired.

Hosted LLM API requests first use Python's system certificate trust. If that TLS verification fails, the Builder tries the independent `certifi` CA bundle, with hostname checking and certificate verification still enabled. It does not retry HTTP errors, rate limits, timeouts, or other network failures. If `SSL_CERT_FILE` or `SSL_CERT_DIR` is explicitly configured, that trust configuration remains authoritative and the bundled fallback is skipped.

If both stores reject the connection:

1. Check the ComfyUI computer's date, time, and timezone.
2. Update `certifi` in the Python environment that runs ComfyUI. For a Windows portable installation, run this from `ComfyUI_windows_portable`:

   ```bat
   python_embeded\python.exe -m pip install --upgrade certifi
   ```

3. Check VPN, proxy, antivirus HTTPS inspection, and any configured custom CA bundle. An expired server or inspection-proxy certificate must be renewed by its operator; updating a CA bundle cannot make an expired certificate valid.
4. Restart ComfyUI after updating dependencies or installing this code change, then retry the LLM Runner action.

The Builder does not disable TLS verification or send API keys over an unverified connection. Downloading or importing unknown certificates is not part of this fix.
