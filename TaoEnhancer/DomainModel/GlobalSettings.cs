using System.ComponentModel.DataAnnotations;
using Common;

namespace DomainModel
{
    public class GlobalSettings
    {
        [Key]
        public int Id { get; set; }
        public bool TestingMode { get; set; }
        public Common.Config.Platform SelectedPlatform { get; set; } = Common.Config.Platform.Windows;
    }
}
