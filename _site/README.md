# Deep Learning 
CMSC 498A: Independent Study, Deep Learning, Fall 2018

# About
The `gh-pages` branch of this repository contains files for the [deep learning blog](http://josephbergman.com/deep-learning). The blog was created using Jekyll and the [Tale](https://github.com/chesterhow/tale) theme. 

# Setup
If you would like to recreate a similar blog follow these steps. 

1. Install Jekyll
```
# Install xcode command line tools 
xcode-select --install

# Make sure you have ruby 2.2.5 or greater 
ruby -v

# Install Jekyll and Bundler 
gem install bundler jekyll
```

2. Create a new repository: `deep-learning` in this case 
3. Create a new branch named `gh-pages` and switch to it 
4. Copy the files from [Tale](https://github.com/chesterhow/tale) into your new repo 
5. Delete `CODE_OF_CONDUCT.md`, `LICENSE`, and `README.md`
6. In `_config.yml` delete the line `base_url: "/tale"`
7. Run `bundle` and `bundle update` if needed 
8. Run `bundle exec jekyll serve`
9. Check your site at http://127.0.0.1:4000
10. When you're done, push it to GitHub 
