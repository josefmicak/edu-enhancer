using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionTemplateStatistics
    {
        [Key]
        public string UserLogin { get; set; } = default!;
        public User User { get; set; } = default!;
        public int SubquestionTemplatesAdded { get; set; } = 0;
        public double NeuralNetworkAccuracy { get; set; } = 0;
    }
}
