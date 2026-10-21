# Exercise - Working with Git

Each working group must implement the functions illustrated
 in the previous lessons regarding the loading of MNIST digit dataset
 and the computation of the centroids using Git tools for shared development.

#### The goal is to pass the automated tests in the project for the `main` branch and push to `release`.

If all the functions have been correctly implemented, you will find a `passed` green badge
 under GitHub Actions. If the tests are failing, a red `failed` badge will be displayed.

In particular each group must develop:
- `load_mnist(csv_filename)`, load the MNIST dataset (available in `data/mnist_data.csv`)
- `split_data(x, y, fract_tr=0.5)`, split data (return 2 sets, training and test)
- `NMC.fit(x, y)`, compute the average centroid for each class for training set
- `NMC.predict(x)`, predict the class of each sample

**All functions must be added to the `fun_utils.py` and `nmc.py` files**,
 which already contain the signature of the required functions.
 
Tests are available inside `tests/test.py` script.

By default the following libraries can be used:
 - numpy
 - scikit-learn
 - matplotlib
 - pandas

Change the `requirements.txt` file by adding the name of any new dependency if necessary.

### Build the documentation

Install the project dependencies, including Sphinx, and run the following
command from the repository root:

```sh
sphinx-build -b html docs docs/build
```

Open `docs/build/index.html` to view the generated documentation.

### Workflow details:
1. The team leader *only* should fork the current project and provide the 
 repository url to other team members. The team leader must also add other team members
 to his project using the Settings -> Members menu of the GitHub project. Members should
 be added with **Maintainer** role. All team members can now clone the project locally.
2. Each group should **discuss** how to develop each function
 (example: if any extra function is needed, what it should return, ...).
3. Each group should then **discuss** which function each team member should implement.
4. Development of each function must be done in a **specific branch**,
 which must be pushed to the remote after creation.
 **No direct commits into the `main` branch should be done**
5. Each team member is suggested to commit local changes to remote
 as often as possible, so that other team members can see the progress
 on the code via the project page.
6. During the development, team members are encouraged to discuss code
 implementation and possible problems.
7. If needed, changes from other feature branches can be merged into a specific feature branch.
 Use any Git tool needed to correctly integrate the changes from multiple branches.
8. After the development of a specific feature has been completed, 
 the relative branch should be merged into `main`.
9. The Activities should now be consulted to see the result of the automated tests.
 Depending on the load of the test runners, the pipeline could show 
 the `pending` status for *quite a while*. Just wait until a runner picked the job.
 Recall that the test scripts can be also run locally.
10. If any change to the code is necessary, it should be developed in the branch
 relative to the specific feature and merged into master to see if tests are then passing.
11. When all tests are passing, the `release` branch should be created from the branch `main`.
 The tests will be run for the `release` branch too and **must succeed**.
12. When all tests are passing **for the `release`** branch, the last commit in that branch
 should be tagged as `v1.0`.


## Details about functions and variables
Common variables:
- y: labels (one for each image) (numpy array)
- x: set of images. numpy array of size (nImages, nFeatures). Each row represents an image.

```python
load_mnist(csv_filename):
    """Load the MNIST dataset.
    
    input:
        csv_filename: string with the path to the dataset to load
    
    output:    
        X: set of images (numpy array)
        y: labels (numpy array)

    """

split_data(x, y, fract_tr):
    """Split the data X,y into two random subsets.
    
    input:
        x: set of images
        y: labels
        fract_tr: float, percentage of samples to put in the training set.
            If necessary, number of samples in the training set is rounded to
            the lowest integer number.
    
    output:
        Xtr: set of images (numpy array, training set)
        Xts: set of images (numpy array, test set)
        ytr: labels (numpy array, training set)
        yts: labels (numpy array, test set)
    
    """


NMC.fit(x, y):
    """Compute the average centroid for each class.

    This function should populate the `._centroids` attribute
    with a numpy array of shape (num_classes, num_features).
    
    input:
        x: set of images (training set, numpy array)
        y: labels (training set, numpy array)
    
    """

NMC.predict(x):
    """Predict the class of each input.
    
    input:
        x: set of images (test set, numpy array)

    output:
        y: labels (numpy array)
    
    """

```
