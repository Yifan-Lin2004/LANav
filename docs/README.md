# LANav project website

Static website and media resources for LANav.

- [Project page](https://yifan-lin2004.github.io/LANav/)
- [Robot video](https://yifan-lin2004.github.io/LANav/video.html)
- [Research code and installation](https://github.com/Yifan-Lin2004/LANav)
- [Paper on arXiv](https://arxiv.org/abs/2607.18794)

GitHub Pages publishes from `main` and the repository root. Two small root HTML
entry points redirect existing project and video links into this directory.
Keep the root `.nojekyll` file. All website assets use relative paths.

Video and manuscript parts are fetched and reconstructed by `media.js`, with
size and SHA-256 checks. The video page supports playback and MP4 download;
the manuscript download link is in the website footer.
