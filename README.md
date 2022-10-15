# Vision Transformer의 모든 것

- [Build Page](https://pseudo-lab.github.io/All-About-ViT)
- [jupyterbook documentation](https://jupyterbook.org/en/stable/intro.html)

## jupyter book build

- install

```
pip install -U jupyter-book
```

- build

```
jupyter-book build book/
```


## publish online

- push

```
git add .
git commit -m "added book!"
git push
```

- publish

```
ghp-import -n -p -f book/_build/html -m "initial publishing"
```
