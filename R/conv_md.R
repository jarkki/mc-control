##  Replaces LaTex equations in a markdown file with .png images of the
##   equations.
##
##  Modified version of https://github.com/muschellij2/latexreadme
## 
##  Run this script from the mc-control/R directory
##
##
library(knitr)
library(stringr)
library(animation)

source("./github_latex.R")

rmd = "../README_.md"
new_md = "../README.md" ## file.path(tempdir(), "README.md")

parse_latex(rmd,
            new_md,
            git_username = "jarkki",
            git_reponame = "mc-control",
            git_branch = "develop",
            git_image_dir = "figures")
##            text_height = 12)

new_html = pandoc(new_md, format = "html")
browseURL(new_html)

