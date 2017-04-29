# MatrixReading.jl
#
# File for reading a sparse SPAM dataset into julia.

module MatrixReading

export ReadMatrix;

# (sp_matrix, tokenlist, category) = ReadMatrix(filename::ASCIIString)
#
# Reads the file stored at 'filename,' which is of the format of
# MATRIX.TEST, and returns a 3-tuple. The first part is 'sp_matrix',
# an m-by-n sparse matrix, where m is the number of training/testing
# examples and n is the dimension, and each row of sp_matrix consists
# of counts of word appearances. (So sp_matrix[i, j] is the number of
# times word j appears in document i.)
#
# tokenlist is a list of the words, where tokenlist[1] is the first
# word in the dictionary and tokenlist[end] is the last.
#
# category is a {0, 1}-valued vector of positive and negative
# examples. Before using in SVM code, you should transform categories
# to have signs +/-1.
function ReadMatrix(filename::ASCIIString)
  fstream = open(filename);
  # Read header line, discard
  headerline = readline(fstream);
  # Read rows and columns, turn into integers
  row_col_line = readline(fstream);
  row_col_split = split(row_col_line);
  num_rows = parse(Int, row_col_split[1]);
  num_cols = parse(Int, row_col_split[2]);
  # Read the list of tokens - just a long string!
  tokenlist = readline(fstream);
  tokenlist = split(tokenlist);

  # Now to read the matrix into the matrix. Each row represents a
  # document (mail), each column represents a distinct token. As the
  # data isn't actually that big, we just use a full matrix to save
  # time.
  full_mat = zeros(num_rows, num_cols);
  categories = zeros(num_rows);
  for ii = 1:num_rows
    str_inds = readline(fstream);
    str_split = split(str_inds);
    categories[ii] = parse(Int, str_split[1]);
    jj = 2;
    ind = 1;
    while (str_split[jj] != "-1")
      ind = parse(Int, str_split[jj]) + ind;
      count = parse(Int, str_split[jj + 1]);
      full_mat[ii, ind] = count;
      jj = jj + 2;
    end
  end
  close(fstream);
  return (sparse(full_mat), tokenlist, categories);
end


end   # module MatrixReading
