# Just a quick converter
#' @title Wrapper for im.convert for PDF to PNG conversion
#'
#' @description This function calls \code{\link{im.convert}} to convert a PDF to a
#' PNG
#' @param pdfname Name of output PDF. Pngname will be same name as pdf but with png extension
#' @param extra.opts Arguments passed to \code{\link{im.convert}}
#' @export
#' @import animation
#' @seealso \code{\link{im.convert}}
#' @return Name of PNG
converter <- function(pdfname, # Name of output PDF.  Pngname will be same name as pdf, but with png extension
                     extra.opts = "-density 300" # Arguments passed to \code{\link{im.convert}}
                     ){
  ####################################################
  # Turn off the animation options
  ####################################################
  aniopts = ani.options()
  ani.options(autobrowse = FALSE)
  ani.options(interval = 0)
  pngname = gsub("[.]pdf$", ".png", pdfname)
  im.convert(pdfname, output = pngname, extra.opts = extra.opts)
  ####################################################
  # Reinstate
  ####################################################
  ani.options(aniopts)
  pngname
}


#' @title Creates a PNG of LaTeX Equation
#'
#' @description This code takes in LaTeX, creates a simple LaTeX document, and then
#' creates a PDF of this, crops it, then converts it to a PNG
#' @param latex single character string of LaTeX code to convert
#' @param packages to be loaded into LaTex using \code{\\usepackage}
#' @param ... Arguments to be passed to \code{\link{converter}}
#' @export
#' @return Name of output file as a PNG
png_latex <- function(
  latex, # single character string of LaTeX code to convert
  packages = c("amsmath", "amsfonts", "amssymb", "graphicx"), ## MY CHANGE HERE!! (added graphicx)
  ...
){
  # latex = latexTranslate(latex)
  packages = paste0("\\usepackage{", packages, "}")
  mystr = c("\\documentclass{article}",
            packages,
            "\\begin{document}",
            "\\thispagestyle{empty}",
            "\\scalebox{0.5}{",           ## MY CHANGE HERE!!
            latex,
            "}",
            "\\end{document}")
  stub = tempfile(fileext = "")
  infile = paste0(stub, ".tex")
  writeLines(mystr, con = infile)
  # system(sprintf("open %s", infile))
  outfile = paste0(stub, ".pdf")
  cwd = getwd()
  setwd(tempdir())
  system(sprintf("pdflatex %s", basename(infile)))
  setwd(cwd)
  plot_crop(outfile)
  converter(outfile, ...)
  outfile = paste0(stub, ".png")
  return(outfile)
}

#' @title Parse LaTeX for Markdown
#'
#' @description This function will take an Rmd file and knit it, or and md file, and any
#' LaTex with $ or $$ delineations, create pngs of these equations, and then insert
#' a link to these equations to display in markdown
#' @param rmd Rmd or md file that needs to be parsed
#' @param new_md Output md file (usually named README.md for gitHub
#' @param git_username Username for repository (GitHub currently)
#' @param git_reponame Repository name
#' @param git_branch Branch of repository if not master
#' @param text_height Height of LaTeX rendered passed ot \code{insert_string}
#' @param insert_string String of HTML to put in LaTeX figure. Must have 2 \%s for sprintf
#' @param raw_git_site Site where to reference figure.
#' \code{img_prefix = file.path(raw_git_site git_username git_reponame git_branch)}.
#' Must reference the figures directly with proper content-type headers
#' @param bad_string String to sub in for dollar signs temporarily (not need to be changed)
#' @param ... Arguments to be passed to \code{\link{png_latex}}, such as LaTeX packages or
#' a pass through argument to \code{\link{converter}}
#' @export
#' @import knitr
#' @import stringr
#' @return Filename of parsed MD file
#' @examples
#' rmd = file.path(tempdir(), "README_unparse.rmd")
#' download.file(
#' "https://raw.githubusercontent.com/muschellij2/Github_Markdown_LaTeX/master/README_unparse.rmd",
#' destfile = rmd, method = "curl")
#' new_md = file.path(tempdir(), "README.md")
#' parse_latex(rmd,
#'             new_md,
#'             git_username = "muschellij2",
#'             git_reponame = "Github_Markdown_LaTeX")
#' library(knitr)
#' new_html = pandoc(new_md, format = "html")
#' browseURL(new_html)
#'
#' new_md = file.path(tempdir(), "README.md")
#' parse_latex(rmd,
#'             new_md,
#'             git_username = "muschellij2",
#'             git_reponame = "Github_Markdown_LaTeX")
#' library(knitr)
#' new_html = pandoc(new_md, format = "html")
#' browseURL(new_html)#'
parse_latex <- function(
  rmd, # Rmd or md file that needs to be parsed
  new_md, # Output md file (usually named README.md for gitHub)
  git_username = NULL, # GitHub username
  git_reponame = NULL, #
  git_branch = "master", #
  git_image_dir= "figures", ## MY CHANGE HERE!! 
  text_height = 20, # Height of LaTeX rendered, passed ot \code{insert_string}
  insert_string =
    paste0('\n<img src="%s%s" ',
           'alt="Equation Fail"',
           'height="', text_height,
           '">\n'),
  # String of HTML to put in LaTeX figure.  Must have 2 %s for sprintf
  raw_git_site = "https://rawgit.com",
  # Site where to reference figure.  \code{img_prefix = file.path(raw_git_site, git_username, git_reponame, git_branch)}.
  #Must reference the figures directly with proper content-type headers
  bad_string = "ZZZZZZZZZZZZZZZ",
  ...
){

  if (is.null(git_username)){
    stop("Git Username must be specified")
  }
  if (is.null(git_reponame)){
    stop("Git Repository name must be specified")
  }
  img_prefix = file.path(raw_git_site,
                         git_username,
                         git_reponame,
                         git_branch,
                         git_image_dir,"")

  ## outdir = dirname(rmd) 
  outdir = file.path(dirname(rmd), git_image_dir) ## MY CHANGE HERE!!
  
  stopifnot(file.exists(rmd))
  ext = strsplit(rmd, "[.]")[[1]]
  ext = toupper(ext[length(ext)])
  tfile = tempfile(fileext = ".md")
  stopifnot(ext %in% c("MD", "RMD"))
  if (ext == "RMD"){
    knit(input = rmd, output = tfile)
  }
  if (ext == "MD"){
    file.copy(from = rmd, to = tfile)
  }
  xmd = md = readLines(tfile)

  md = paste(md, collapse = "\n")
  double_latex = gsub("\\$\\$(.+?)\\$\\$",
                      paste0(bad_string, "$$\\1$$", bad_string),
                      md)
  double_latex = strsplit(double_latex, bad_string)[[1]]
  double_latex = double_latex[grepl("$$", double_latex, fixed=TRUE)]

  eq_no = 0
  if (length(double_latex) > 0){
    outfiles = sapply(double_latex, png_latex, ...)
    filenames = sprintf("eq_no_%02.0f.png",
                        seq(eq_no+1, eq_no + length(outfiles)))
    eq_no = eq_no + length(outfiles)
    filenames = file.path(outdir, filenames)
    mapply(function(x, y){
      file.copy(x, y, overwrite = TRUE)
    }, outfiles, filenames)

    new_str = sprintf(insert_string, img_prefix,
                      basename(filenames))
    for (istr in seq(length(outfiles))){
      md = sub("\\$\\$(.+?)\\$\\$",
               new_str[istr],
               md)
    }

    writeLines(md, con = tfile)
    xmd = readLines(tfile)
  }

  md = xmd
  ########################
  # Find Code and Remove
  ########################
  start_ticks = grep("^```", md)
  rm.ind = NULL
  if (length(start_ticks) > 0){
    start_ticks = matrix(start_ticks, ncol = 2, byrow = TRUE)
    rm.ind = unlist(apply(start_ticks, 1, function(x){
      seq(x[1], x[2])
    }))
  }
  if (length(rm.ind) > 0 ){
    md = md[-rm.ind]
  }

  ########################
  # Replace single dollar sign inline equations
  ########################
  double_latex = gsub("\\$(.+?)\\$", paste0(bad_string, "$\\1$", bad_string),
                      md)
  double_latex = strsplit(double_latex, bad_string)
  double_latex = unlist(sapply(double_latex, function(x){
    x = str_trim(x[grepl("$", x, fixed = TRUE)])
  }))
  #### turn all into equations
  #   double_latex = sapply(double_latex, function(x){
  #     x = paste0("$", x, "$")
  #   })
  #
  if (length(double_latex) > 0){
    outfiles = sapply(double_latex, png_latex, ...)
    filenames = sprintf("eq_no_%02.0f.png",
                        seq(eq_no+1, eq_no + length(outfiles)))
    eq_no = eq_no + length(outfiles)
    inline_link_filenames <- filenames  ## MY CHANGE HERE!!
    ##inline_links <- cat("![](", git_image_dir, "/", fname, "?raw=true)", sep="") ## MY CHANGE HERE!!
    filenames = file.path(outdir, filenames)
    mapply(function(x, y){
      file.copy(x, y, overwrite = TRUE)
    }, outfiles, filenames)

    md = xmd
    ########################
    # Find Code and sub out the dollar signs for bad_string
    ########################
    start_ticks = grep("^```", md)
    rm.ind = NULL
    if (length(start_ticks) > 0){
      start_ticks = matrix(start_ticks, ncol = 2, byrow = TRUE)
      rm.ind = unlist(apply(start_ticks, 1, function(x){
        seq(x[1], x[2])
      }))
    }
    if (length(rm.ind) > 0 ){
      md[rm.ind] = gsub("\\$", bad_string, md[rm.ind])
    }

    ########################
    # Loop through and put in HTML
    ########################
    md = paste(md, collapse = "\n")
    ## new_str = sprintf(insert_string, img_prefix, ## MY CHANGE HERE!!
    ##                   basename(filenames))
    new_str = sprintf("![](%s/%s?raw=true)", git_image_dir, inline_link_filenames) ## MY CHANGE HERE!!
    for (istr in seq(length(outfiles))){
      md = sub("\\$(.+?)\\$",
               new_str[istr],
               md)
    }

    ########################
    # Put back dollar signs for bad_string
    ########################
    md = gsub(bad_string, "$", md)

    writeLines(md, con = tfile)
    xmd = readLines(tfile)
  }

  file.copy(tfile, new_md, overwrite = TRUE)
  return(new_md)
  #
}
