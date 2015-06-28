import bnpy

if __name__ == '__main__':
    import BarsK10V900
    Data = BarsK10V900.get_data(nDocTotal=500)
    DataIterator = Data.to_iterator(nLap=1, nBatch=10, dataorderseed=8541952)
    while DataIterator.has_next_batch():
        Dbatch = DataIterator.get_next_batch()
        Dbatch.WriteToFile_ldac('batch%02d.ldac' % (DataIterator.batchID))
    Dbatch.writeWholeDataInfoFile('Info.conf')
