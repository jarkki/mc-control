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
            git_image_dir = "figures",
            text_height = 16)

## new_html = pandoc(new_md, format = "html")
## browseURL(new_html)

