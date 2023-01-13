using Common;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionTemplateStatistics
    {
        [Key]
        public string UserLogin { get; set; } = default!;
        public User User { get; set; } = default!;
        public bool EnoughSubquestionTemplatesAdded { get; set; } = false;
        public int SubquestionTemplatesAddedCount { get; set; } = 0;
        public double NeuralNetworkAccuracy { get; set; } = 0;
        public double MachineLearningAccuracy { get; set; } = 0;
        public EnumTypes.Model UsedModel { get; set; } = 0;
    }
}
