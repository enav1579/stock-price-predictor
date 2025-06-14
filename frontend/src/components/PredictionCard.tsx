import { useQuery } from 'react-query';
import { Paper, Typography, Box, CircularProgress, LinearProgress } from '@mui/material';
import axios from 'axios';

interface PredictionCardProps {
  ticker: string;
}

const PredictionCard = ({ ticker }: PredictionCardProps) => {
  const { data, isLoading, error } = useQuery(
    ['prediction', ticker],
    async () => {
      const response = await axios.get(`http://localhost:8000/stocks/predict/${ticker}`);
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
        <Typography color="error">Error loading prediction</Typography>
      </Paper>
    );
  }

  if (!data) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography>No prediction available</Typography>
      </Paper>
    );
  }

  const confidence = Math.round(data.confidence * 100);

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Price Prediction
      </Typography>
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" color="text.secondary">
          Current Price
        </Typography>
        <Typography variant="h5">
          ${data.current_price?.toFixed(2)}
        </Typography>
      </Box>
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" color="text.secondary">
          Predicted Price
        </Typography>
        <Typography variant="h5" color={data.predicted_change > 0 ? 'success.main' : 'error.main'}>
          ${data.predicted_price?.toFixed(2)}
          <Typography component="span" variant="body2" sx={{ ml: 1 }}>
            ({data.predicted_change?.toFixed(2)}%)
          </Typography>
        </Typography>
      </Box>
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Prediction Confidence
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <LinearProgress 
            variant="determinate" 
            value={confidence} 
            sx={{ flexGrow: 1 }}
          />
          <Typography variant="body2">
            {confidence}%
          </Typography>
        </Box>
      </Box>
      <Box>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Feature Importance
        </Typography>
        {data.features_importance && Object.entries(data.features_importance).map(([feature, importance]) => (
          <Box key={feature} sx={{ mb: 1 }}>
            <Typography variant="body2" sx={{ mb: 0.5 }}>
              {feature}
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={importance * 100} 
              sx={{ height: 4, borderRadius: 2 }}
            />
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default PredictionCard; 