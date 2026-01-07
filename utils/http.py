def is_local_request(request) -> bool:
    try:
        host = (request.get_host() or '').split(':', 1)[0].strip().lower()
    except Exception:
        host = ''
    remote = (request.META.get('REMOTE_ADDR') or '').strip().lower()
    origin = (request.META.get('HTTP_ORIGIN') or '').strip().lower()
    referer = (request.META.get('HTTP_REFERER') or '').strip().lower()

    local_hosts = ('localhost', '127.0.0.1', '::1')
    if host in local_hosts or remote in local_hosts:
        return True
    if origin.startswith('http://localhost') or origin.startswith('http://127.0.0.1'):
        return True
    if referer.startswith('http://localhost') or referer.startswith('http://127.0.0.1'):
        return True
    return False
