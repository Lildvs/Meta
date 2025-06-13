![PyPI version](https://badge.fury.io/py/forecasting-tools.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/forecasting-tools.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
[![Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/Dtq4JNdXnw)
[![PyPI Downloads](https://static.pepy.tech/badge/forecasting-tools/month)](https://pepy.tech/projects/forecasting-tools)
[![PyPI Downloads](https://static.pepy.tech/badge/forecasting-tools)](https://pepy.tech/projects/forecasting-tools)

# Overview

This repository contains forecasting and research tools built with Python and Streamlit. The project aims to assist users in making predictions, conducting research, and analyzing data related to hard to answer questions (especially those from Metaculus).

Here are the tools most likely to be useful to you:
- 🎯 **Forecasting Bot:** General forecaster that integrates with the Metaculus AI benchmarking competition and provides a number of utilities. You can forecast with a pre-existing bot or override the class to customize your own (without redoing all the aggregation/API code, etc)
- 🔌 **Metaculus API Wrapper:** for interacting with questions and tournaments
- 📊 **Benchmarking:** Randomly sample quality questions from Metaculus and run your bot against them so you can get an early sense of how your bot is doing by comparing to the community prediction and expected baseline scores.


Here are some other features of the project:
- **Smart Searcher:** A custom AI-powered internet-informed llm powered by Exa.ai and GPT. It is more configurable than Perplexity AI, allowing you to use any AI model, instruct the AI to decide on filters, get citations linking to exact paragraphs, etc.
- **Key Factor Analysis:** Key Factors Analysis for scoring, ranking, and prioritizing important variables in forecasting questions
- **Base Rate Researcher:** for calculating event probabilities (still experimental)
- **Niche List Researcher:** for analyzing very specific lists of past events or items (still experimental)
- **Fermi Estimator:** for breaking down numerical estimates (still experimental)
- **Monetary Cost Manager:** for tracking AI and API expenses
- **Other experimental tools:** See the demo site for other AI forecasting tools that this project supports (not all are documented)

All the examples below are in a Jupyter Notebook called `README.ipynb` which you can run locally to test the package (make sure to run the first cell though).

Join the [discord](https://discord.gg/Dtq4JNdXnw) for updates and to give feedback (btw feedback is very appreciated, even just a quick "I did/didn't decide to use tool X for reason Y, though am busy and don't have time to elaborate" is helpful to know)

Note: This package is still in an experimental phase. The goal is to keep the package API fairly stable, though no guarantees are given at this phase especially with experimental tools. There will be special effort to keep the ForecastBot and TemplateBot APIs consistent.
