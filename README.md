# cs330-homework-2-solved
**TO GET THIS SOLUTION VISIT:** [CS330 Homework 2 Solved](https://www.ankitcodinghub.com/product/cs330-sunet-id-solved-4/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;113023&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS330 Homework 2 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Name:

Overview

In this assignment we will experiment with two meta-learning algorithms, model-agnostic meta-learning (MAML) [1] and prototypical networks [2], for few-shot classification. You will

(1) Implement and train MAML. You also need to experiment with different choices of the key hyperparameter of the MAML algorithm, the inner gradient step size, and implement a variant of MAML that learns the inner step size automatically.

(2) Implement and train prototypical networks.

Similar to Homework 1, we will work with the Omniglot dataset [3], which contains 1623 different characters from 50 different languages. For each character there are 20 28×28 images. We are interested in training models for K-shot, N-way classification.

Submission: To submit your homework, submit one pdf report and one zip file to GradeScope, where the report will contain answers to the deliverables listed below and the zip file contains your code (run maml.py, models/maml.py, run ProtoNet.py, models/ProtoNet.py, load data.py) with the filled in solutions.

Code Overview: The code consists of five main files:

• load data.py: code to load batches of images and labels, which does not need to be edited.

• run maml.py: training and testing script for running MAML.

• models/maml.py: network architecture and computation graph construction of MAML.

• run ProtoNet.py: training and testing script for running prototypical networks.

• models/ProtoNet.py: network architecture/loss of prototypical networks.

There is also the omniglot resized folder with the data. You should not modify this folder.

Dependencies: We expect code in Python 3.5+ with Pillow, scipy, numpy, tensorflow installed. If you use Anaconda, run conda env create -f environment.yml to build a virtual environment containing packages required for the homework.

Figure 1: For each task i, MAML computes inner gradient updates on training datapoints Ditr and evaluates the loss on test datapoints Dits. Averaging over all tasks, the outer loop loss function is optimized w.r.t. the original model parameter θ to learn an initialization that can quickly adapt to new tasks during meta-test time.

Note: Even though convolutional networks will be used for MAML and Prototypical Networks, the code should be able to be run with CPU.

Problem 1: Model-Agnostic Meta-Learning (MAML) [1]

We will first attempt few-shot classification with MAML. As introduced in the class, during meta-training phase, MAML operates in two loops, an inner loop and an outer loop. In the inner loop, MAML computes gradient updates using examples from each task and calculates the loss on test examples from the same task using the updated model parameters. In the outer loop, MAML aggregates the per-task post-update losses and performs a meta-gradient update on the original model parameters. At meta-test time, MAML computes new model parameters based a few examples from an unseen class and uses the new model parameters to predict the label of a test example from the same unseen class. The main idea of MAML is shown in Figure 1. The data processing will be done in the same way as in Homework 1. In the run maml.py and models/maml.py files:

1. Fill in the data processing parts in the meta train and meta test functions in run maml.py, which should call the data generator provided in load data.py to generate a batch of images and their corresponding labels. You should partition the batch into two parts, inputa, labela and inputb, labelb, where inputa, labela are used to compute gradient updates in the inner loop and inputb, labelb are used to get the task losses after the gradient update. Hint: You need to fill in data processing parts for metatraining, meta-validation and meta-test, but they should be fairly similar.

2. Fill in the function called task inner loop in the models/maml.py file which takes inputs inputa, labela, inputb, labelb and computes the inner loop updates in the main MAML algorithm. Feel free to use self.loss func to compute the losses. Your main work should be calculating the gradient updates of each weight variable stored in the weights dictionary and passing the updated weights to forward conv in

Figure 2: Prototypical networks compute the prototypes of all tasks using training datapoints Ditr. Then by comparing the query example x to each of the prototype, the model makes prediction based on the softmax function over the distance between the embedding of the query and all prototypes.

3. Run python run maml.py –n way=5 –k shot=1 –inner update lr=0.4 –num inner updates=1. Also try with inner update lr being 0.04 and 4.0. For each configuration, submit a plot of the validation accuracy over iterations as well as the number of the average test accuracy. Can you briefly explain why different values of inner update lr would affect meta-training?

4. Tuning inner update lr could turn out to be tricky when running MAML for different datasets. A variant of MAML [4] proposes to automatically learn the inner update lr. Try to learn separate inner update lr per num inner update per weight variable. Specifically, for each inner gradient update, for each weight variable stored in the weights dictionary, initialize one inner update lr variable and learn it using backpropagation. Plot the meta-validation accuracy over meta-training iterations and state how it compares to the MAML with fixed inner update lr.

Problem 2: Prototypical Networks [2]

1. Similar to Problem 1, fill in the data processing parts in run ProtoNet.py, which should also call the data generator provided in load data.py. You should partition the sampled batch into support, i.e. the per-task training data, and query, i.e. the per-task test datapoints. The support will be used to calculate the prototype of each class and query will be used to compute the distance to each prototype. You also need to get labels of the query examples in order to compute the cross-entropy loss for training the whole model.

2. Fill in the function called ProtoLoss in the models/ProtoNet.py file which takes the embeddings of the support and query examples as well as the one-hot label encodings of the queries and computes the loss and prediction accuracy based on the main algorithm of the prototypical networks.

3. Run python run ProtoNet.py ./omniglot resized/ and plot the validation accuracy over iterations. Report the average test accuracy along with its standard deviation.

Problem 3: Comparison and Analysis

After implementing both meta-learning algorithms, we would like to compare them a bit. In practice, we usually have limited amount of meta-training data but relatively more meta-test datapoints. Hence one interesting comparison would be meta-training both algorithms with 5-way 1-shot regime but meta-testing them using 4-shot data. Specifically, run the following to evaluate Prototypical Networks:

python run ProtoNet.py ./omniglot resized/ –n-way=5 –k-shot=1 –n-query=5

–n-meta-test-way=5 –k-meta-test-shot=4 –n-meta-test-query=4

For evaluating MAML, first do meta-training by running:

python run maml.py –n way=5 –k shot=1 –inner update lr=0.4

–num inner updates=1

Then restore the weights to get meta-test performance by running:

python run maml.py –n way=5 –k shot=4 –inner update lr=0.4

–num inner updates=1 –meta train=False –meta test set=True –meta train k shot=1

Try K = 4,6,8,10 at meta-test time. Compare the meta-test performance between MAML and Prototypical Networks by plotting the meta-test accuracies over different choices of K.

References

[3] Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. Human-level concept learning through probabilistic program induction. Science, 350(6266):1332–1338, 2015.
