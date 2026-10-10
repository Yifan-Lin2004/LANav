"use strict";

const copyButton = document.querySelector("#copy-citation");
const copyStatus = document.querySelector("#copy-status");
const citation = document.querySelector("#bibtex");

function selectCitation() {
  const selection = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(citation);
  selection.removeAllRanges();
  selection.addRange(range);
}

if (copyButton && copyStatus && citation) {
  copyButton.addEventListener("click", async () => {
    const text = citation.textContent.trim() + "\n";
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch (_) {
      // Fall through to the selection-based copy method.
    }
    if (!copied) {
      selectCitation();
      try {
        copied = document.execCommand("copy");
      } catch (_) {
        copied = false;
      }
    }
    if (copied) {
      copyStatus.textContent = "BibTeX copied to clipboard.";
      copyButton.querySelector("span").textContent = "Copied";
      setTimeout(() => { copyButton.querySelector("span").textContent = "Copy BibTeX"; }, 2200);
    } else {
      selectCitation();
      copyStatus.textContent = "Citation selected. Press Ctrl+C or ⌘C to copy, or download the .bib file.";
    }
  });
}
