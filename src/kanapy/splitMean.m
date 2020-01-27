function mo = splitMean(o,n)

    olen = length(o);

    nn = histcounts(1:olen,n);
    or = mat2cell(o,nn);
    orm = cellfun(@mean,or,'UniformOutput',false); 
    mo = [orm{:}];

end