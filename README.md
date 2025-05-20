# NZ_chatbot

Steps to be flowed in the respective order for execution in a CLI:
1. Select a suitable directory to run the chatbot.
2. Clone this GitHub repository
3. python -m venv myenv  #you can give an appropriate name to your environment
4. Activate your environment. source myenv/bin/activate.
5. Time to install the requirements. pip install -r requirements.txt
6. Make we have all the information in our knowledge repository. Let's scrap the text fields if that are missing. Here is how to do it:
     i) Run the file scrap.py followed by the path of the knowledge repository. Like this :
    ii) python scrap.py /mnt/e/Downloads/NZ_AI_CHALLENGE_2025/NZ_AI_CHALLENGE_2025/wikipedia_sample_with_10_percent_batch_text_null.csv
   iii) This will save a new csv file with the scrapped details in the same directory with the name 'NZ_repository_scrapped_1.csv'.
    iv) Make sure this new file is in the same directory as the code base. I have already scrapped the missing text fields so I have already have a NZ_repository_scrapped_1.csv file in my directory.
        You can use the same file that I have created.
7. The rest of the code is modularized. All you need to run is : python main.py --interactive
8. As the knowledge repository is huge, there is additional optional parameter that can be passed after --interactive. It determines how many chunks of the data will be considered.
     i) if you want all the data to be considered, pass the argument 'all' like this : python main.py --interactive all
        This is time consuming and it takes about 3 hrs to convert the entire repository into embeddings.
    ii) To just test the chatbot we can start of with a small number like 50 or 100, like this:  python main.py --interactive 50
        This will take 1-2 mins to process.
   iii) If no argument is passed, the default value is set to a 1000, which take approximately 4 mins to process.
