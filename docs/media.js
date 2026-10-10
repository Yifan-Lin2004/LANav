(() => {
  "use strict";

  const mediaCache = new Map();
  let notice;

  function showNotice(message, fallback) {
    if (!notice) {
      notice = document.createElement("p");
      notice.className = "media-notice";
      notice.setAttribute("role", "status");
      notice.setAttribute("aria-live", "polite");
      document.body.append(notice);
    }
    notice.replaceChildren(document.createTextNode(message));
    if (fallback) {
      const link = document.createElement("a");
      link.href = fallback;
      link.textContent = " Open the resource.";
      notice.append(link);
    }
    notice.hidden = false;
  }

  function loadMedia(path, onProgress) {
    const manifestUrl = new URL(path, document.baseURI).href;
    let entry = mediaCache.get(manifestUrl);
    if (!entry) {
      entry = { listeners: new Set(), progress: 0 };
      mediaCache.set(manifestUrl, entry);
      entry.promise = (async () => {
        const response = await fetch(manifestUrl);
        if (!response.ok) throw new Error("Media information is unavailable.");
        const manifest = await response.json();
        if (!manifest || typeof manifest.filename !== "string" ||
            typeof manifest.type !== "string" ||
            !Number.isSafeInteger(manifest.bytes) || manifest.bytes <= 0 ||
            !Array.isArray(manifest.parts) || !manifest.parts.length ||
            manifest.parts.some(part => !part || typeof part.path !== "string" ||
              !Number.isSafeInteger(part.bytes) || part.bytes <= 0) ||
            manifest.parts.reduce((sum, part) => sum + part.bytes, 0) !== manifest.bytes) {
          throw new Error("Media information is invalid.");
        }

        const buffers = new Array(manifest.parts.length);
        let next = 0;
        let loadedBytes = 0;
        async function worker() {
          while (next < manifest.parts.length) {
            const index = next++;
            const part = manifest.parts[index];
            const partResponse = await fetch(new URL(part.path, manifestUrl));
            if (!partResponse.ok) throw new Error("The media could not be loaded.");
            const buffer = await partResponse.arrayBuffer();
            if (buffer.byteLength !== part.bytes) throw new Error("The media is incomplete.");
            buffers[index] = buffer;
            loadedBytes += buffer.byteLength;
            entry.progress = Math.floor(loadedBytes / manifest.bytes * 100);
            for (const listener of entry.listeners) listener(entry.progress);
          }
        }
        await Promise.all(Array.from({ length: Math.min(4, manifest.parts.length) }, worker));

        const blob = new Blob(buffers, { type: manifest.type });
        if (blob.size !== manifest.bytes) throw new Error("The media is incomplete.");
        if (manifest.sha256 && window.crypto && window.crypto.subtle) {
          const digest = await window.crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
          const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
          if (hash !== manifest.sha256.toLowerCase()) throw new Error("The media could not be verified.");
        }
        return { url: URL.createObjectURL(blob), filename: manifest.filename, bytes: blob.size };
      })().catch(error => {
        mediaCache.delete(manifestUrl);
        throw error;
      });
    }
    if (onProgress) {
      entry.listeners.add(onProgress);
      onProgress(entry.progress);
    }
    return entry.promise.finally(() => {
      if (onProgress) entry.listeners.delete(onProgress);
    });
  }

  document.querySelectorAll("video[data-video-manifest]").forEach(async video => {
    const status = document.createElement("p");
    status.className = "media-status";
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    video.closest(".video-frame").after(status);
    video.setAttribute("aria-busy", "true");
    try {
      const media = await loadMedia(video.dataset.videoManifest, progress => {
        status.textContent = progress < 100 ? `Loading video… ${progress}%` : "Preparing video…";
      });
      video.src = media.url;
      video.load();
      status.textContent = "Video ready. Use the controls to play.";
      video.dispatchEvent(new CustomEvent("mediaready", { detail: { bytes: media.bytes, filename: media.filename } }));
    } catch (_) {
      status.textContent = "The video could not be loaded. Reload the page to try again.";
    } finally {
      video.removeAttribute("aria-busy");
    }
  });

  document.querySelectorAll("a[data-media-download]").forEach(link => {
    link.addEventListener("click", async event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      if (link.getAttribute("aria-busy") === "true") return;
      link.setAttribute("aria-busy", "true");
      try {
        const media = await loadMedia(link.dataset.mediaDownload, progress => {
          showNotice(progress < 100 ? `Preparing download… ${progress}%` : "Preparing download…");
        });
        const download = document.createElement("a");
        download.href = media.url;
        download.download = media.filename;
        document.body.append(download);
        download.click();
        download.remove();
        showNotice("Download ready.");
        setTimeout(() => { if (notice.textContent === "Download ready.") notice.hidden = true; }, 5000);
      } catch (_) {
        showNotice("The download could not be prepared. Please try again.", link.href);
      } finally {
        link.removeAttribute("aria-busy");
      }
    });
  });
})();
