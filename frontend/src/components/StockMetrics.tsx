import { useQuery } from 'react-query';
import { Paper, Typography, Grid, CircularProgress } from '@mui/material';
import axios from 'axios';

interface StockMetricsProps {
  ticker: string;
}

const StockMetrics = ({ ticker }: StockMetricsProps) => {
  const { data, isLoading, error } = useQuery(
    ['metrics', ticker],
    async () => {
      const response = await axios.get(`http://localhost:8000/stocks/metrics?tickers=${ticker}`);
      return response.data[0];
    }
  );

  if (isLoading) {
    return (
      <Paper sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <CircularProgress />
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography color="error">Error loading metrics</Typography>
      </Paper>
    );
  }

  if (!data) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography>No data available</Typography>
      </Paper>
    );
  }

  const metrics = [
    { label: 'Price', value: data.price?.toFixed(2) },
    { label: 'Change', value: `${data.change?.toFixed(2)}%` },
    { label: 'P/E Ratio', value: data.pe?.toFixed(2) },
    { label: 'EPS', value: data.eps?.toFixed(2) },
    { label: 'ROE', value: data.roe?.toFixed(2) },
    { label: 'ROA', value: data.roa?.toFixed(2) },
    { label: 'Market Cap', value: formatMarketCap(data.market_cap) },
    { label: 'Volume', value: formatNumber(data.volume) },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Current Metrics
      </Typography>
      <Grid container spacing={2}>
        {metrics.map((metric) => (
          <Grid item xs={6} key={metric.label}>
            <Typography variant="subtitle2" color="text.secondary">
              {metric.label}
            </Typography>
            <Typography variant="body1">
              {metric.value || 'N/A'}
            </Typography>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

const formatMarketCap = (value?: number) => {
  if (!value) return 'N/A';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toFixed(2)}`;
};

const formatNumber = (value?: number) => {
  if (!value) return 'N/A';
  return value.toLocaleString();
};

export default StockMetrics; 