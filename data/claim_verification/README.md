## Data Organization
This directory holds the contents of claim verification subset of **ArCOV-19**. These are the components:

- claims: tab-separated file that stores full information for each claim including the claim's: ID, text, veracity label, topical category and online source from which we extract the claim.
- relevant_tweets: each file in this directory contains the IDs of relevant tweets and their veracity labels for a claim. 
- propagation_networks: contains the IDs of retweets and conversational threads for the relevant tweets to each claim. 
