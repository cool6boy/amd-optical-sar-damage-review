"""Bounded HTTP-range ZIP reader. Never silently falls back to full download."""
import io
import re
import time
import urllib.request
import zipfile


class ByteBudget:
    def __init__(self, limit):
        self.limit = int(limit)
        self.received = 0

    def charge(self, count):
        self.received += count
        if self.received > self.limit:
            raise RuntimeError("Network byte budget exceeded")


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, budget, timeout=15):
        super().__init__()
        self.url, self.size, self.budget, self.timeout = url, int(size), budget, timeout
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if value < 0:
            raise ValueError("Negative seek")
        self.pos = value
        return value

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.pos
        count = min(count, max(0, self.size - self.pos))
        if count == 0:
            return b""
        if count > 64 * 1024 * 1024 or self.budget.received + count > self.budget.limit:
            raise RuntimeError("Range too large for configured budget")
        start, end = self.pos, self.pos + count - 1
        # Distinct query prevents a proxy returning a cached range for another offset.
        sep = "&" if "?" in self.url else "?"
        url = self.url + sep + "adr_range=" + str(start) + "-" + str(end)
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "amd-disaster-review/0.1"})
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status != 206 or not match or tuple(map(int, match.groups())) != (start, end, self.size):
                        raise ValueError("Server did not honor exact HTTP Range; refusing full download")
                    data = response.read(count + 1)
                    self.budget.charge(len(data))
                    if len(data) != count:
                        raise ValueError("Incomplete range")
                self.pos += count
                return data
            except (TimeoutError, OSError):
                if attempt:
                    raise
                time.sleep(0.5)


class RemoteZip:
    def __init__(self, url, size, budget):
        self.file = RangeFile(url, size, budget)
        self.zip = zipfile.ZipFile(self.file)

    def names(self):
        return self.zip.namelist()

    def read(self, name, max_uncompressed=32 * 1024 * 1024):
        info = self.zip.getinfo(name)
        if info.file_size > max_uncompressed:
            raise ValueError("ZIP member exceeds expansion limit")
        # zipfile verifies CRC as part of read.
        return self.zip.read(name)

    def find_basename(self, basename):
        matches = [n for n in self.names() if n.rsplit("/", 1)[-1] == basename]
        if len(matches) != 1:
            raise ValueError(f"Expected unique archive member: {basename}; got {len(matches)}")
        return matches[0]

    def close(self):
        self.zip.close()
        self.file.close()
