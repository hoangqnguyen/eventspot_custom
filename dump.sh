# this dumps all the python files in the current directory and its subdirectories
# and then prints the contents of the files using batcat
# could be useful for chatgpt prompts

tree -P "*.py" -I "__pycache__" -f | tee dump.txt | grep ".py$" | awk '{print $NF}' | while read file; do
  echo "$file" >> dump.txt
  batcat "$file" >> dump.txt
done
