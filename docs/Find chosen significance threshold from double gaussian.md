## Find chosen significance threshold from double gaussian
Can represent a gaussian as:

```python
def gaussian(x,A,u,sigma):
    return A * scipy.stats.norm.pdf((x-u)/sigma) / sigma
```

Can get the cdf as:
  
```python
def cdf_gaussian(x,A,u,sigma):
    return A * scipy.stats.norm.cdf(x,loc=u,scale=sigma)
```

From this should be able to get the cdf of two gaussians together.

One could then solve numerically for the $x$ value of the desired percentile.

$$\displaystyle (1 - \frac{p}{N}) = \frac{cdf(x,A_1,u_1,\sigma_1) + cdf(x,A_2,u_2,\sigma_2)}{A_1 + A_2}$$

Where $p = 0.05$ is the probability threshold (~ p-value) of mislabelling 1 background point as a galaxy, $N$ is the number of pixels under consideration

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>