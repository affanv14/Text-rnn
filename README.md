# Text-rnn [WIP]
  Word/Character rnn in tensorflow

    * Input is given through text.txt
    * Change rnn config in config.py
    * Default mode is for training characters

#Usage

  To run the rnn with given inputfile
  ~~~
  python main.py -i <input file>
  ~~~

  To generate data with the trained rnn
  ~~~
  python main.py -g
  ~~~


# OPTIONS
    -h,--help             print help text and exit
    -q,--quiet            Dont print anything on screeen
    -i,--input            specify input file
    -n,--new              start from scratch and overwrite previous save file
    -u,--summary          generate summary
    -s,--save             specify save file(default is provided)
    -g,--generate         generate data
    -w,--word             trains rnn for word data
