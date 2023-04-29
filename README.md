# Telegram Bot BasilAI

## Install

- prod environment

    ```bash
    make install-prod
    ```

- dev environment

    ```bash
    make install-dev
    ```

## Generate dataset

```bash
make dataset RAW_DATA=<path to telegram dialogs dump json> TARGET=<user_id> DATASET=<path to csv file to save generated dataset>
```

## Generate advanced dataset

```bash
make dataset_advanced \
  USER_ID=<user id> \
  RAW_DATASET=<path to raw json dialog file> \
  DIALOG_SECONDS_COOLDOWN=<minimum time distance in seconds between different dialogs> \
  DIALOG_MEMORY=<message number in memory to answer> \
  DATASET_OUTPUT=<path to save generated dataset folder>
```

## Fit tokenizer

```bash
make fit_tokenizer \
  TOKENIZER_DATASET=<path to the dataset> \
  TOKENIZER=<tokenizer class> \
  TOKENIZER_SIZE=<tokenizer vocab size> \
  TOKENIZER_OUTPUT=<path to save tokenizer>
```

## Train model

```bash
make train TRAIN_CONFIG=<path to config file>
```

## Run telegram bot locally

```bash
make bot TOKEN=<telegram bot auth token>
```

## Heroku hosting

### Getting started with heroku

- authenticate on heroku

  ```bash
  heroku login
  ```

- add git `heroku` remote

  ```bash
  heroku git:remote -a basilai-bot
  ```
  
### Pushing new version to heroku

- push local `master` branch to heroku `main` branch

  ```bash
  git push heroku master:main
  ```
  
  > _you can push any local branch - just change `master` to branch name_

- heroku restarts application with new version automatically
  
### Scaling heroku application

- check heroku running apps

  ```bash
  heroku ps
  ```

- scale heroku apps

  ```bash
  heroku ps:scale worker=1
  ```

- stop heroku apps

  ```bash
  heroku ps:stop worker.1
  ```

### Get logs

```bash
heroku logs -t
```