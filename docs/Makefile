# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile prepare_examples

# Custom target to copy example notebooks into the source directory
prepare_examples:
	@echo "Copying example notebooks to docs/source/examples..."
	@mkdir -p "$(SOURCEDIR)/examples"
	@cp -r ../examples "$(SOURCEDIR)/"

# Catch-all target: route all unknown targets to Sphinx.
# This now depends on 'prepare_examples' to ensure notebooks are copied first.
%: Makefile prepare_examples
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
