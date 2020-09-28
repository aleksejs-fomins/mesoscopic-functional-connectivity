function rez_arr = cell_to_array(cell_arr_3D, offDiag)

  dim12 = size(cell_arr_3D);
  dim3  = size(cell_arr_3D{1,2}{1});
  dim123 = [dim12 dim3(2)];

  rez_arr = zeros(dim123);
  
  for i = 1:dim123(1)
      for j = 1:dim123(2)
          if (i ~= j) || (offDiag == 0)
              rez_arr(i, j, :) = cell_arr_3D{i, j}{1};
          end
      end
  end
end

