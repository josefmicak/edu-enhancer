using Common;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionResultStatistics
    {
        [Key]
        public string UserLogin { get; set; } = default!;
        public User User { get; set; } = default!;
        public int SubquestionResultsAdded { get; set; } = 0;
        public double NeuralNetworkAccuracy { get; set; } = 0;
        public double MachineLearningAccuracy { get; set; } = 0;
        public EnumTypes.Model UsedModel { get; set; } = 0;
    }
}
