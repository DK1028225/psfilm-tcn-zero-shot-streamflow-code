"""Synthetic tests of the Stage16/Stage18 frozen data protocol."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import data_pipeline as dp


class TestFrozenData(unittest.TestCase):
    def make_df(self, basin, n=80):
        d = {col: np.arange(n, dtype=float) * (j+1) / 100. for j,col in enumerate(dp.RAIN_COLUMNS[basin])}
        q = np.linspace(3, 20, n, dtype=float)
        q[2] = -1.0
        q[6] = np.nan
        d['流量(m³/s)'] = q
        d['sample_time'] = pd.date_range('2018-01-01', periods=n, freq='h').astype(str)
        return pd.DataFrame(d).loc[:,['sample_time']+list(dp.RAIN_COLUMNS[basin])+['流量(m³/s)']]

    def setUp(self):
        self.basin = '安墩河'

    def test_preprocessing_invalid_time_retains_prior_row_effect(self):
        import tempfile
        d = self.make_df(self.basin)
        d.loc[7,'sample_time'] = 'invalid'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'synthetic.csv';d.to_csv(p,index=False)
            result,audit=dp.preprocess_unified_df(p,self.basin)
        self.assertEqual(len(result),79)
        self.assertEqual(audit['raw_negative_flow_count'],1)
        self.assertEqual(int(result['流量_原始负值指示'].sum()),1)
        self.assertEqual(int(result['流量_缺失指示'].sum()),1)
        self.assertEqual(float(result.iloc[2]['流量(m³/s)']),0.0)
        # At original row 8, the lag is still constructed from original row 7.
        expected=np.log1p(float(d.loc[7,'流量(m³/s)'])/dp.BASIN_PHYSICS[self.basin][0]*1000)
        self.assertAlmostEqual(float(result.iloc[7]['Log前期径流模数']),expected,places=10)

    def test_split_and_dataset_record_positions(self):
        raw=self.make_df(self.basin)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'synthetic.csv';raw.to_csv(p,index=False)
            df,_=dp.preprocess_unified_df(p,self.basin)
        tr,val,te=dp.split_source_df(df)
        self.assertEqual(tuple(map(len,(tr,val,te))),(56,12,12))
        scx,scy,scs,_=dp.fit_fold_scalers({self.basin:tr},[self.basin])
        ds=dp.PhysicsHydroDataset(df,self.basin,0,24,scx,scy,scs)
        self.assertEqual(len(ds),56)
        item=ds[0]
        self.assertEqual(len(item),7)
        self.assertEqual(tuple(item[0].shape),(24,2))
        self.assertEqual(tuple(item[1].shape),(4,))
        self.assertEqual(float(item[3]),float(df.loc[23,'流量_缺失指示']))
        self.assertAlmostEqual(float(item[5]),float(df.loc[22,'流量(m³/s)']),places=6)
        self.assertEqual(item[6],str(pd.to_datetime(df.loc[23,'时间'])))

    def test_lobo_sampler_equal_total_mass(self):
        raw={b:self.make_df(b,240+i*15) for i,b in enumerate(dp.LOBO_BASIN_ORDER)}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data={}
            for b,df in raw.items():
                path=Path(tmp)/(str(dp.LOBO_BASIN_ORDER.index(b))+'.csv')
                df.to_csv(path,index=False)
                data[b]=dp.preprocess_unified_df(path,b)[0]
        result=dp.build_fold_data(data,heldout_basin='增江',seed=42)
        totals=result['sampling_audit']['total_basin_sampler_weight'].to_numpy()
        np.testing.assert_allclose(totals,np.ones(5),atol=1e-10)
        self.assertEqual(result['train_loader'].sampler.replacement,True)
        self.assertEqual(result['train_loader'].sampler.num_samples,len(result['train_loader'].dataset))
        self.assertNotIn('增江',result['sources'])
        self.assertIsNotNone(result['heldout_dataset'])
        self.assertEqual(len(result['heldout_dataset']),len(data['增江'])-24)
        self.assertIsNotNone(result['heldout_loader'])

    def test_predefined3_preserves_per_sample_weights(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data={}
            for b in dp.ORIGINAL_SOURCE_BASINS:
                raw=self.make_df(b,n=240+15*dp.LOBO_BASIN_ORDER.index(b))
                path=Path(tmp)/(b+'.csv');raw.to_csv(path,index=False)
                data[b]=dp.preprocess_unified_df(path,b)[0]
        result=dp.build_fold_data(data,source_basins=dp.ORIGINAL_SOURCE_BASINS,seed=42)
        np.testing.assert_allclose(result['sampling_audit']['per_sample_weight'].to_numpy(),[1.0,0.5,1.0])
        self.assertEqual(result['train_loader'].sampler.replacement,True)
        self.assertEqual(result['train_loader'].sampler.num_samples,len(result['train_loader'].dataset))

    def test_static_scaler_source_only(self):
        b1='安墩河';b2='船塘河'
        f1=self.make_df(b1,90)
        f2=self.make_df(b2,100)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data={}
            for b,raw in [(b1,f1),(b2,f2)]:
                path=Path(tmp)/(b+'.csv');raw.to_csv(path,index=False)
                data[b]=dp.preprocess_unified_df(path,b)[0]
        tr={b:dp.split_source_df(v)[0] for b,v in data.items()}
        x,y,s,ss=dp.fit_fold_scalers(tr,[b1,b2])
        np.testing.assert_allclose(s.mean_,np.mean(np.array([dp.BASIN_PHYSICS[b1],dp.BASIN_PHYSICS[b2]],dtype=np.float32),axis=0))

if __name__=='__main__':
    unittest.main()
