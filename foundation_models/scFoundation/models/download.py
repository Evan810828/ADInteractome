import gdown

url = 'https://drive.google.com/uc?id=1N3HFbQIi1ZznfxExj3wRe0aa_GxOVIZM'
output = 'model.ckpt'
gdown.download(url, output, quiet=False)
