---
title: Cleaning Step Data
author: ~
date: '2019-09-24'
slug: cleaning-step-data
categories: []
tags: []
description: ''
featured: ''
featuredalt: ''
featuredpath: ''
linktitle: ''
---

My most recent paper that I published is titled, [*"Cardiometabolic thresholds for peak 30-min cadence and steps/day."*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0219933)  In the article we used multiple methods to establish thresholds for step activity.  I was able to work with several outstanding and established faculty at West Point and other universities.  The data was collected by NHANES using cluster sampling.  This was my first time analyzing data of this nature, but fortunately there are several resources available.  My favorite part of the analysis was cleaning the data.  Here is the code that I used to clean the data.  The raw data set is called `steps`.

```{r}
CleanedSteps<-steps %>%
  group_by(SEQN) %>%
  mutate(check = cumsum(c(F, abs(diff(PAXSTEP)) > 0))) %>%
  group_by(check, add = TRUE) %>%
  filter(!(PAXSTEP == 0 & n() >= 60))%>%
  ungroup %>%
  mutate(PAXSTEP=ifelse(PAXINTEN<=500,0,PAXSTEP))%>%
  filter(PAXCAL==1,PAXSTAT==1)%>%
  select(-check)%>%
  group_by(SEQN,PAXDAY)%>%
  arrange(desc(PAXSTEP))%>%
  filter(PAXSTEP<=180)%>%
  summarise(Total.Day.Ware.Time=n(),Peak1=PAXSTEP[1],
            Peak30=mean(PAXSTEP[1:30]),Peak60=mean(PAXSTEP[1:60]),
            Zero.Time=sum(PAXSTEP==0), Band.One.Time=sum(0<PAXSTEP & PAXSTEP<=19),
            Band.Two.Tim=sum(19<PAXSTEP & PAXSTEP<=39),
            Band.Three.Time=sum(39<PAXSTEP & PAXSTEP<=59),
            Band.Four.Time=sum(59<PAXSTEP & PAXSTEP<=79),
            Band.Five.Time=sum(79<PAXSTEP & PAXSTEP<=99),
            Band.Six.Time=sum(99<PAXSTEP & PAXSTEP<=119),
            Band.Seven.Time=sum(120<=PAXSTEP),
            Total.Steps.Per.Day=sum(PAXSTEP))
```
There are several rules that are used to summarize the steps, and still much more debate going on in different circles.  The majority of the rules were developed by Dr. Catrine Tudor-Locke.  My favorite part of the code was finding 60 consecutive minutes of non-ware time.  To do this I had to use a mix of `dplyr` and `base` commands.  The following portion of the code accomplishes identifying and removing non-ware time.