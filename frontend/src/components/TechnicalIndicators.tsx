import { useQuery } from 'react-query';
import { Paper, Typography, Grid, CircularProgress } from '@mui/material';
import axios from 'axios';

interface TechnicalIndicatorsProps {
  ticker: string;
}

const TechnicalIndicators = ({ ticker }: TechnicalIndicatorsProps) => {
  const { data, isLoading, error } = useQuery(
    ['indicators', ticker],
    async () => {
      const response = await axios.get(`http://localhost:8000/stocks/indicators/${ticker}`);
      return response.data;
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
        <Typography color="error">Error loading indicators</Typography>
      </Paper>
    );
  }

  if (!data?.indicators) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography>No indicators available</Typography>
      </Paper>
    );
  }

  const indicators = [
    { label: 'MA20', value: data.indicators.ma20?.toFixed(2) },
    { label: 'MA50', value: data.indicators.ma50?.toFixed(2) },
    { label: 'MA200', value: data.indicators.ma200?.toFixed(2) },
    { label: 'RSI', value: data.indicators.rsi?.toFixed(2) },
    { label: 'MACD', value: data.indicators.macd?.toFixed(2) },
    { label: 'Signal Line', value: data.indicators.signal_line?.toFixed(2) },
  ];

  const getRSIColor = (value?: number) => {
    if (!value) return 'inherit';
    if (value > 70) return 'error.main';
    if (value < 30) return 'success.main';
    return 'inherit';
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Technical Indicators
      </Typography>
      <Grid container spacing={2}>
        {indicators.map((indicator) => (
          <Grid item xs={6} key={indicator.label}>
            <Typography variant="subtitle2" color="text.secondary">
              {indicator.label}
            </Typography>
            <Typography 
              variant="body1"
              color={indicator.label === 'RSI' ? getRSIColor(Number(indicator.value)) : 'inherit'}
            >
              {indicator.value || 'N/A'}
            </Typography>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default TechnicalIndicators; 