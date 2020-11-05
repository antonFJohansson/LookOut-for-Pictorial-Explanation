from sklearn.ensemble import IsolationForest
import copy
import numpy as np

class LookOut():

  """
  Class that implements the algorithm LookOut ARTICLE LINK HERE
  """

  def fit(self, X):

    """
    Fits the outlier detector to the data and detects the outliers
    Args:
      X: The input data (numpy array, num_data_points \times num_features)
    """
    self.X = X
    

    clf = IsolationForest(random_state=0).fit(X)
    scores = clf.score_samples(X)
    outliers = clf.predict(X)

    self.sorted_ind = scores.argsort()
    scores = -scores[self.sorted_ind]
    outliers = outliers[self.sorted_ind]

    self.outlier_ind = np.where(outliers==-1)[0]
    self.outlier_scores = scores[self.outlier_ind]
  
    self.feat_pair_list = [(i,j) for i in range(self.X.shape[1]) for j in range(i+1, self.X.shape[1])]
    self.s_matrix = np.zeros((len(self.outlier_scores),len(self.feat_pair_list)))

    for p_id, (feat_1, feat_2) in enumerate(self.feat_pair_list):

      curr_X = X[:,(feat_1, feat_2)]
      clf = IsolationForest(random_state=0).fit(curr_X)
      scores = -clf.score_samples(curr_X)
      curr_scores = scores[self.sorted_ind[self.outlier_ind]]
      self.s_matrix[:,p_id] = curr_scores

  def calc_p_score(self, set_of_plots):

    """ Calculates the score for a given set of plots """
      
    if len(set_of_plots) == 0:
      p_score = 0
    elif len(set_of_plots) == 1:
      p_score = np.sum(self.s_matrix[0:self.num_outliers, tuple(set_of_plots)])
    else:
      p_score = np.sum(np.max(self.s_matrix[0:self.num_outliers, tuple(set_of_plots)], axis = 1))

    return p_score

  def plot(self, plot_budget = 3, num_outliers = None, axis_names = None):

    """
    Plots the results of the algorithm.
    Args:
      plot_budget: The number of plots to consider (Int)
      num_outliers: Number of top most outliers to consider (Int)
      axis_names: List with the names for each axis
    """

    self.axis_names = axis_names

    ## How many outliers to display
    if num_outliers is None:
      self.num_outliers = self.s_matrix.shape[0]
    else:
      self.num_outliers = num_outliers
    
    if num_outliers > self.s_matrix.shape[0]:
          raise ValueError('You have specified to showcase more outliers than what has been detected. Num detected outliers: {}'.format(self.s_matrix.shape[0]))
    if plot_budget > len(self.feat_pair_list):
        raise ValueError('You have specified to showcase more plots than what is available. Num available plots: {}'.format(len(self.feat_pair_list)))

    set_of_plots = []

    for bbb in range(plot_budget):
      available_plots = [jjj for jjj in range(len(self.feat_pair_list)) if jjj not in set_of_plots]

      best_marg_gain = -1
      ## Find the best plots
      for p in available_plots:

        new_set_of_plots = copy.copy(set_of_plots)
        new_set_of_plots.append(p)
        marg_gain = self.calc_p_score(new_set_of_plots) - self.calc_p_score(set_of_plots)
        if marg_gain > best_marg_gain:
          best_marg_gain = marg_gain
          best_set = copy.copy(new_set_of_plots)

      set_of_plots = copy.copy(best_set)
    ## Detect which outlier should belong to which plot
    o_p_ids = np.argmax(self.s_matrix[0:self.num_outliers,tuple(set_of_plots)], axis = 1)
    best_feat_comb = [self.feat_pair_list[i] for i in set_of_plots]

    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(1, len(best_feat_comb), figsize = (8*plot_budget,8))
    for idx,(f1, f2) in enumerate(best_feat_comb):
      
      curr_data = self.X[:, (f1,f2)]
      if len(best_feat_comb) > 1:
          ax[idx].scatter(curr_data[:,0], curr_data[:,1])
      else:
          ax.scatter(curr_data[:,0], curr_data[:,1])
      curr_outlier_ind = np.where(o_p_ids==idx)[0]
      curr_outlier_ind = self.outlier_ind[curr_outlier_ind]
      curr_outliers = curr_data[self.sorted_ind[curr_outlier_ind],:]
      if len(best_feat_comb) > 1:
          ax[idx].scatter(curr_outliers[:,0], curr_outliers[:,1])
      else:
          ax.scatter(curr_outliers[:,0], curr_outliers[:,1])
      
      if self.axis_names is None:
        if len(best_feat_comb) > 1:
            ax[idx].set_xlabel('Feature {}'.format(f1))
            ax[idx].set_ylabel('Feature {}'.format(f2))
        else:
            ax.set_xlabel('Feature {}'.format(f1))
            ax.set_ylabel('Feature {}'.format(f2))
      else:
        #print(f1, f2)
        #print(self.axis_names)
        #print(self.axis_names[f1], self.axis_names[f2])
        if len(best_feat_comb) > 1:
            ax[idx].set_xlabel(self.axis_names[f1])
            ax[idx].set_ylabel(self.axis_names[f2])
        else:
            ax.set_xlabel(self.axis_names[f1])
            ax.set_ylabel(self.axis_names[f2])
      ## Add the index of the point to the plot
      for pt_id in self.sorted_ind[curr_outlier_ind]:
        if len(best_feat_comb) > 1:
            ax[idx].annotate(str(pt_id), (curr_data[pt_id,0], curr_data[pt_id,1]))
        else:
            ax.annotate(str(pt_id), (curr_data[pt_id,0], curr_data[pt_id,1]))
    
