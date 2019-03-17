# Character Segmenter

Segment and separate characters in a bitmap image

Simple utility to segment and output separate characters in an image containing text. Expects the characters to be "not much vertically overlapping". There is, however, a configurable maximum expected text lenght that, if exceeded, will cause the algorithm to try harder to separate characters.

# Setting up

Get [Pipenv](https://pipenv.readthedocs.io) for Python 3.7, then:

```
pipenv install
```

Trying out character segmenting visually with a sample file:

```
pipenv run python display_segments.py test_data/word_tonally.png
```

# Developing

Set up a Git pre-commit hook to run integration tests before commits:

```
git config core.hooksPath .githooks
```
