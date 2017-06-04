# **Solving LSAT Logic Games** ##

## Objective
The Law School Admissions Test (LSAT) includes a section on [analytical reasoning](http://www.lsac.org/jd/lsat/prep/analytical-reasoning) that requires test takers to solve logic puzzles.  My Capstone project solves some of these puzzles and answers LSAT questions about them.

## Motivation
Within NLP, semantic parsing or natural language understanding remains largely an unsolved problem.  Aside from corpuses too vast for humans to read comfortably, the reading comprehension skills of the best NLP systems typically lag far behind those of humans.  I hope to demonstrate that, even on a tiny corpus where size poses no challenge to a human reader, algorithms can sometimes outperform humans.

## Data
My data consists of questions and answers from actual LSAT examinations, which are copyrighted.  I may also incorporate simulated LSAT questions constructed by test prep companies.

## References
[_Foundations of Statistical Natural Language Processing_](https://nlp.stanford.edu/fsnlp/)  
Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze

[_Speech and Language Processing_](https://web.stanford.edu/~jurafsky/slp3/)  
Dan Jurafsky and James Martin

[_Introduction to Information Retrieval_](https://nlp.stanford.edu/IR-book/)  
Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze

[Natural Language Toolkit: Comparative Sentence Corpus Reader](http://www.nltk.org/_modules/nltk/corpus/reader/comparative_sents.html)
* Nitin Jindal and Bing Liu. "Identifying Comparative Sentences in Text Documents".
   Proceedings of the ACM SIGIR International Conference on Information Retrieval
   (SIGIR-06), 2006.

* Nitin Jindal and Bing Liu. "Mining Comprative Sentences and Relations".
   Proceedings of Twenty First National Conference on Artificial Intelligence
   (AAAI-2006), 2006.

* Murthy Ganapathibhotla and Bing Liu. "Mining Opinions in Comparative Sentences".
    Proceedings of the 22nd International Conference on Computational Linguistics
    (Coling-2008), Manchester, 18-22 August, 2008.

## Courses
[Natural Language Processing (Columbia)](http://www.cs.columbia.edu/~cs4705/)  
Michael Collins

[Natural Language Processing with Deep Learning (Stanford)](http://web.stanford.edu/class/cs224n/)  
Richard Socher and Chris Manning

##  Steps of the Challenge
To answer questions about logic games, five steps must be performed.  Here, I describe each step, indicate its degree of difficulty, and suggest how I intend to approach it:

1. Classify the Puzzle: _medium_  
I will use standard text classification tools from SpaCy and Sci-Kit Learn, perhaps aided by feature engineering, to determine what type of puzzle the prompt represents.

2. Set-Up: _medium_  
I will use SpaCy's entity recognizer to extract the names of the relevant entities from the puzzle prompt.  The possible permutations/combinations of these names define the event space for the puzzle--that is, the possible solutions to the puzzle.

3. Parse Rules: _very difficult_  
The number of possible permutations shrinks as rules are introduced that limit the permissible permutations.  To parse these rules, I will construct a Chomsky Formal Grammar and write a parser that translates the LSAT's English statements of the rules into executable functions.  

4. Apply the Rules to Narrow the Possible Solutions: _medium_  
The rules are then applied to all the events in the puzzle's sample space to reduce the set of possible solutions to the puzzle.

5. Parse Questions: _medium_  
The questions must be parsed to determine what is being asked.  Some questions impose additional, local rules, which further reduce the number of possible solutions.  These local rules must be parsed as in Step 3.

6. Parse Multiple Choice Answers: _easy_  
The test's five possible answers must be parsed.

7. Select Answer: _easy_  
From the five answers, the correct answer must be identified.
