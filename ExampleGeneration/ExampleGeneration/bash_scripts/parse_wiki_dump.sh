cd WIKI_DUMP_DIR

yourfilenames=`ls ./*.bz2`
echo $yourfilenames
cd ..

counter=0
for eachfile in $yourfilenames
do
   echo "starting, first unzipping"
   echo $eachfile

   unziped_file_name=${eachfile%."bz2"}
   unziped_file_name_parsed=${unziped_file_name:2}

   cd wiki_dump_1
   bzip2 -dk $eachfile
   cd ..


   echo "finished unzipping"
   echo $unziped_file_name_parsed

   unziped_file="wiki_dump_1/${unziped_file_name_parsed}"
   output_file="OUTPUT_DIR${unziped_file_name_parsed}_parsed.gz"

   echo $unziped_file
   echo $output_file

   chmod 777 $unziped_file

   echo "running script for file"

   python ExampleGeneration/ExampleGeneration/run.py -c ParseWikiDump -in $unziped_file -out $output_file

   echo "removing file"

   rm $unziped_file

   cp "wiki_dump_1/${unziped_file_name_parsed}.bz2" "finished_script/${unziped_file_name_parsed}.bz2"

   echo "finished file"

done